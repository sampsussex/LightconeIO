#!/bin/env python
#
# Low-memory rework of the lightcone particle -> halo matching script.
#
# Main changes relative to the original:
#
#  1. BATCHED PROCESSING: particle files are processed in batches grouped by
#     file_nr (i.e. by shell/redshift), so peak memory is set by
#     --files-per-batch instead of by the total lightcone size. The halo
#     catalogue is pre-filtered per batch to the comoving distance range the
#     batch's particles can actually reach (min AND max bounds).
#
#  2. float32 POSITIONS: particle and halo positions are cast to float32
#     (~60 pc absolute precision at 1 Gpc -- negligible vs halo radii).
#     This halves the dominant arrays and all sort/exchange traffic.
#
#  3. NO PER-PARTICLE HaloMass IN PASS 1: the pass-1 "HaloMass" dataset was
#     redundant -- pass 2 already copies BoundSubhalo/TotalMass onto the
#     particles via IndexInHaloLightcone. Mass is now only tracked internally
#     when the overlap method needs it.
#
#  4. CHUNKED KDTREES + VECTORISED QUERIES: instead of one KDTree over all
#     local particles, trees are built over chunks of --tree-chunk-size
#     particles (bounding tree memory), and halos are queried in batches with
#     query_ball_point(positions, radii, workers=-1) instead of a per-halo
#     Python loop.
#
#  5. CHEAPER ORDER RESTORATION: the final "restore original particle order"
#     step is a direct alltoallv scatter computed from the known original
#     partitioning, instead of a second full parallel sort.
#
#  6. FIXED MASS_WEIGHTED METHOD: the original criterion cancelled itself out
#     (the r^2 factor appeared on both sides) and referenced uninitialised
#     -1 masses. It now assigns each particle to the halo minimising
#     r^2 / M_halo, as the comment originally intended.
#
#  7. MEMORY REPORTING: message() prints rank-0 peak RSS; report_memory()
#     does a collective max/sum across ranks at key points.
#
# Pass 2 (copying halo properties onto particles) is also batched. Consider
# trimming `halo_properties` -- e.g. Lightcone/HaloCentre costs 24 bytes per
# particle on disk and anything joinable later via SOAPIndex can be dropped.

import os
import sys
import argparse
import time
import gc
import resource
t0 = time.time()

import numpy as np
import h5py
import scipy.spatial

from mpi4py import MPI
comm = MPI.COMM_WORLD
comm_size = comm.Get_size()
comm_rank = comm.Get_rank()

import virgo.util.match as match
import virgo.mpi.parallel_hdf5 as phdf5
import virgo.mpi.parallel_sort as psort
import virgo.mpi.util as mpi_util


# Constants to identify methods for dealing with particles in multiple halos
FRACTIONAL_RADIUS = 0
MOST_MASSIVE      = 1
LEAST_MASSIVE     = 2
MASS_WEIGHTED     = 3
overlap_methods = {
    "fractional-radius" : FRACTIONAL_RADIUS,
    "most-massive"      : MOST_MASSIVE,
    "least-massive"     : LEAST_MASSIVE,
    "mass-weighted"     : MASS_WEIGHTED,
    }


def rank_peak_rss_gb():
    # ru_maxrss is in KB on Linux
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0**2)


def message(m):
    if comm_rank == 0:
        elapsed = time.time() - t0
        print(f"{elapsed:.1f}s [rank0 peak RSS {rank_peak_rss_gb():.1f}GB]: {m}",
              flush=True)


def report_memory(label):
    """Collective: report max and total peak RSS across all ranks."""
    rss = rank_peak_rss_gb()
    max_rss = comm.allreduce(rss, op=MPI.MAX)
    sum_rss = comm.allreduce(rss, op=MPI.SUM)
    message(f"Memory [{label}]: max rank peak RSS = {max_rss:.1f}GB, "
            f"sum over ranks = {sum_rss:.1f}GB")


def read_lightcone_halo_positions_and_radii(args, radius_name, mass_name):
    """
    Read in the lightcone halo catalogue and cross reference with SOAP
    to find the radius and mass for each halo in the lightcone.

    Assumes that positions in the lightcone are comoving and in the
    same units as SOAP (except for the expansion factor dependence).
    """

    message("Reading lightcone halo catalogue")
    halo_lightcone_datasets = ("Lightcone/HaloCentre",
                               "Lightcone/SnapshotNumber",
                               "InputHalos/HaloCatalogueIndex",
                               "InputHalos/SOAPIndex")
    mf = phdf5.MultiFile(args.halo_lightcone_filenames,
                         file_idx=np.arange(args.first_snap_nr, args.final_snap_nr+1),
                         comm=comm)
    halo_lightcone_data = mf.read(halo_lightcone_datasets, group="/", read_attributes=True)

    # Positions in float32: at ~1 Gpc box scales this is ~60 pc absolute
    # precision, negligible compared to halo radii. Halves memory/traffic.
    halo_lightcone_data["Lightcone/HaloCentre"] = \
        np.ascontiguousarray(halo_lightcone_data["Lightcone/HaloCentre"], dtype=np.float32)

    # Store index in halo lightcone of each halo
    nr_local_halos = len(halo_lightcone_data["InputHalos/HaloCatalogueIndex"])
    offset = comm.scan(nr_local_halos) - nr_local_halos
    halo_lightcone_data["IndexInHaloLightcone"] = np.arange(nr_local_halos, dtype=np.int64) + offset

    # Repartition halos for better load balancing
    message("Repartition halo catalogue")
    nr_local_halos = len(halo_lightcone_data["InputHalos/HaloCatalogueIndex"])
    nr_total_halos = comm.allreduce(nr_local_halos)
    nr_desired = np.zeros(comm_size, dtype=int)
    nr_desired[:] = nr_total_halos // comm_size
    nr_desired[:nr_total_halos % comm_size] += 1
    assert np.sum(nr_desired) == nr_total_halos
    for name in halo_lightcone_data:
        halo_lightcone_data[name] = psort.repartition(halo_lightcone_data[name], nr_desired, comm=comm)

    # The input catalogue is ordered by redshift, but we want a mix of redshifts on each rank
    message("Reassign halos to MPI ranks")
    nr_local_halos = len(halo_lightcone_data["InputHalos/HaloCatalogueIndex"])
    rng = np.random.default_rng()
    sort_key = rng.integers(comm_size, size=nr_local_halos, dtype=np.int32)
    order = psort.parallel_sort(sort_key, comm=comm, return_index=True)
    for name in sorted(halo_lightcone_data):
        psort.fetch_elements(halo_lightcone_data[name], order, result=halo_lightcone_data[name], comm=comm)

    # Sort locally by snapnum
    message("Sorting local lightcone halos by snapshot")
    order = np.argsort(halo_lightcone_data["Lightcone/SnapshotNumber"])
    for name in halo_lightcone_data:
        halo_lightcone_data[name] = halo_lightcone_data[name][order,...]

    # Find range of local halos at each snapshot
    message("Identifying halos at each snapshot")
    unique_snap, snap_offset, snap_count = np.unique(halo_lightcone_data["Lightcone/SnapshotNumber"],
                                                     return_index=True, return_counts=True)

    # Find full range of snapshots across all MPI ranks
    min_snap = comm.allreduce(np.amin(unique_snap), op=MPI.MIN)
    max_snap = comm.allreduce(np.amax(unique_snap), op=MPI.MAX)

    # Make snapnum, count and offset arrays which include snapshots not
    # present on this rank: we do collective reads of the SOAP outputs so all
    # ranks need to agree on what range of snapshots to do.
    nr_snaps = max_snap - min_snap + 1
    unique_snap_all = np.arange(min_snap, max_snap+1, dtype=int)
    snap_offset_all = np.zeros(nr_snaps, dtype=int)
    snap_count_all  = np.zeros(nr_snaps, dtype=int)
    for us, so, sc in zip(unique_snap, snap_offset, snap_count):
        i = us - min_snap
        assert unique_snap_all[i] == us
        snap_offset_all[i] = so
        snap_count_all[i] = sc

    # Allocate storage for radius of each lightcone halo
    nr_halos = len(halo_lightcone_data["InputHalos/HaloCatalogueIndex"])
    halo_lightcone_data[radius_name] = None # Don't know dtype for radius array yet

    # Loop over snapshots
    for snapnum in unique_snap_all:

        soap_datasets = ("InputHalos/HaloCatalogueIndex", radius_name, mass_name)

        # Read the SOAP catalogue for this snapshot
        message(f"Reading SOAP output for snapshot {snapnum}")
        mf = phdf5.MultiFile(args.soap_filenames % {"snap_nr" : snapnum}, file_idx=(0,), comm=comm)
        soap_data = mf.read(soap_datasets, read_attributes=True)

        # Get the expansion factor of this snapshot
        if comm_rank == 0:
            with h5py.File(args.soap_filenames % {"snap_nr" : snapnum}, "r") as infile:
                a = float(infile["SWIFT"]["Header"].attrs["Scale-factor"])
        else:
            a = None
        a = comm.bcast(a)

        # Ensure radii are in comoving units
        radius_a_exponent = float(soap_data[radius_name].attrs["a-scale exponent"])
        soap_data[radius_name] *= a**(radius_a_exponent-1.0)

        # Optional search radius multiplier (was hardcoded 2x)
        if args.radius_multiplier != 1.0:
            soap_data[radius_name] *= args.radius_multiplier
        if comm_rank == 0:
            message(f"Using radius {radius_name} x {args.radius_multiplier}")

        # Match lightcone halos at this snapshot to SOAP halos: the matching
        # has already been done upstream, we just use InputHalos/SOAPIndex.
        message("Finding lightcone halos in SOAP output")
        i1 = snap_offset_all[snapnum-min_snap]
        i2 = snap_offset_all[snapnum-min_snap] + snap_count_all[snapnum-min_snap]
        assert np.all(halo_lightcone_data["Lightcone/SnapshotNumber"][i1:i2] == snapnum)

        ptr = halo_lightcone_data["InputHalos/SOAPIndex"][i1:i2]
        assert np.all(ptr >= 0) # All halos in the lightcone should be found in SOAP

        # Allocate storage for radii now that we know what dtype SOAP uses
        if halo_lightcone_data[radius_name] is None:
            radius_dtype = soap_data[radius_name].dtype
            halo_lightcone_data[radius_name] = phdf5.AttributeArray(-np.ones(nr_halos, dtype=radius_dtype),
                                                                    attrs=soap_data[radius_name].attrs)
            mass_dtype = soap_data[mass_name].dtype
            halo_lightcone_data[mass_name] = phdf5.AttributeArray(-np.ones(nr_halos, dtype=mass_dtype),
                                                                  attrs=soap_data[mass_name].attrs)

        message("Storing radii for lightcone halos at this snapshot")
        psort.fetch_elements(soap_data[radius_name], ptr,
                             result=halo_lightcone_data[radius_name][i1:i2], comm=comm)

        message("Storing masses for lightcone halos at this snapshot")
        psort.fetch_elements(soap_data[mass_name], ptr,
                             result=halo_lightcone_data[mass_name][i1:i2], comm=comm)

        del soap_data
        gc.collect()

    # All halos should have been assigned a radius
    assert np.all(halo_lightcone_data[radius_name] >= 0)
    assert np.all(halo_lightcone_data[mass_name] >= 0)

    report_memory("after reading halo lightcone")
    return halo_lightcone_data


def read_lightcone_index(args):
    """
    Read the index file, determine which particle types are present, and
    build BATCHES of particle files grouped by file_nr (i.e. by shell /
    redshift), so each batch spans a narrow comoving-distance range and the
    halo catalogue can be pre-filtered per batch.

    Returns:
      type_z_range     : dict of (min_z, max_z) per particle type
      file_batches     : list of lists of input particle filenames
    """

    # Particle types which may be in the lightcone
    type_names = ("BH", "Gas", "Stars")

    type_z_range = {}

    index_file = args.lightcone_dir+"/"+args.lightcone_base+"_index.hdf5"
    if comm_rank == 0:
        with h5py.File(index_file, "r") as index:
            lc = index["Lightcone"]
            nr_mpi_ranks = int(lc.attrs["nr_mpi_ranks"])
            final_file_on_rank = lc.attrs["final_particle_file_on_rank"]
            for tn in type_names:
                min_z = float(lc.attrs["minimum_redshift_"+tn])
                max_z = float(lc.attrs["maximum_redshift_"+tn])
                if max_z > min_z:
                    type_z_range[tn] = (min_z, max_z)
    else:
        nr_mpi_ranks = None
        final_file_on_rank = None
        type_z_range = None
    nr_mpi_ranks, final_file_on_rank, type_z_range = comm.bcast((nr_mpi_ranks, final_file_on_rank, type_z_range))

    # Report which particle types we found
    for name in type_z_range:
        min_z, max_z = type_z_range[name]
        message(f"have particles for type {name} from z={min_z} to z={max_z}")

    # Group files by file_nr so that each group covers the same shells on
    # every writing rank, then accumulate whole groups into batches of at
    # least args.files_per_batch files.
    max_file_nr = int(np.max(final_file_on_rank))
    file_batches = []
    current = []
    for file_nr in range(max_file_nr+1):
        group = []
        for rank_nr in range(nr_mpi_ranks):
            if file_nr <= final_file_on_rank[rank_nr]:
                filename = (f"{args.lightcone_dir}/{args.lightcone_base}_particles/"
                            f"{args.lightcone_base}_{file_nr:04d}.{rank_nr}.hdf5")
                group.append(filename)
        current.extend(group)
        if len(current) >= args.files_per_batch:
            file_batches.append(current)
            current = []
    if len(current) > 0:
        file_batches.append(current)

    nr_files_total = sum(len(b) for b in file_batches)
    message(f"Have {nr_files_total} particle files in {len(file_batches)} batches")

    return type_z_range, file_batches


def restore_particle_order(part_index, data_arrays, entry_counts):
    """
    Return data_arrays reordered back to the partitioning described by
    entry_counts (the per-rank counts when part_index was created as a
    contiguous global range). This is a single alltoallv scatter instead of
    a full parallel sort: the destination rank of every element is known
    directly from its index.
    """
    entry_counts = np.asarray(entry_counts, dtype=np.int64)
    rank_offsets = np.cumsum(entry_counts) - entry_counts
    my_offset = rank_offsets[comm_rank]
    my_count = entry_counts[comm_rank]

    # Sort locally by global index: destinations become contiguous
    order = np.argsort(part_index)
    part_index = part_index[order]
    dest = np.searchsorted(rank_offsets, part_index, side="right") - 1
    send_count = np.bincount(dest, minlength=comm_size).astype(np.int64)
    send_offset = np.cumsum(send_count) - send_count
    recv_count = np.asarray(comm.alltoall(send_count), dtype=np.int64)
    recv_offset = np.cumsum(recv_count) - recv_count
    assert np.sum(recv_count) == my_count

    # Exchange the indexes so receivers know where to place elements
    idx_recv = np.empty(my_count, dtype=part_index.dtype)
    psort.my_alltoallv(part_index, send_count, send_offset,
                       idx_recv, recv_count, recv_offset, comm=comm)
    local_pos = idx_recv - my_offset
    del idx_recv

    results = []
    for arr in data_arrays:
        arr_sorted = arr[order]
        recv = np.empty(my_count, dtype=arr.dtype)
        psort.my_alltoallv(arr_sorted, send_count, send_offset,
                           recv, recv_count, recv_offset, comm=comm)
        del arr_sorted
        out = np.empty_like(recv)
        out[local_pos] = recv
        del recv
        results.append(out)
    return results


def compute_particle_group_index(halo_id, halo_pos, halo_radius, halo_mass, part_pos,
                                 overlap_method, tree_chunk_size, halo_query_batch):
    """
    Tag particles which are within the search radius of a halo.

    Returns (part_halo_id, part_halo_r_frac) in the input particle order.
    Halo input arrays are not modified (rebound to filtered copies only).
    """

    # Record the partitioning at entry so we can restore it at the end
    nr_particles = part_pos.shape[0]
    entry_counts = np.asarray(comm.allgather(nr_particles), dtype=np.int64)
    offset = np.cumsum(entry_counts)[comm_rank] - nr_particles
    part_index = np.arange(nr_particles, dtype=np.int64) + offset

    nr_halos = halo_pos.shape[0]
    nr_halos_total = comm.allreduce(nr_halos)
    nr_particles_total = comm.allreduce(nr_particles)
    message(f"Have {nr_particles_total} particles and {nr_halos_total} halos")

    # Early out: nothing to do for this batch/type
    if nr_particles_total == 0:
        return (-np.ones(nr_particles, dtype=np.int64),
                -np.ones(nr_particles, dtype=np.float32))

    # Split the halos and particles by x coordinate, with a roughly constant
    # number of particles per rank. First, sort the particles by x.
    message("Sorting particles by x coordinate")
    sort_key = part_pos[:,0].copy()
    order = psort.parallel_sort(sort_key, return_index=True, comm=comm)
    del sort_key
    psort.fetch_elements(part_pos, order, result=part_pos, comm=comm)
    psort.fetch_elements(part_index, order, result=part_index, comm=comm)
    del order

    # Find the maximum halo radius
    local_max_radius = float(np.amax(halo_radius)) if halo_radius.size > 0 else -np.inf
    max_radius = comm.allreduce(local_max_radius, op=MPI.MAX)

    # Find min/max distance to any particle (guard against empty ranks)
    if nr_particles > 0:
        part_dist = np.sqrt(np.sum(part_pos.astype(np.float64)**2, axis=1))
        local_max_particle_distance = float(np.amax(part_dist))
        local_min_particle_distance = float(np.amin(part_dist))
        del part_dist
    else:
        local_max_particle_distance = -np.inf
        local_min_particle_distance = np.inf
    max_particle_distance = comm.allreduce(local_max_particle_distance, op=MPI.MAX)
    min_particle_distance = comm.allreduce(local_min_particle_distance, op=MPI.MIN)

    # Keep only halos which can overlap this batch's particle shell:
    # bounded from BOTH sides now that we process narrow distance ranges.
    halo_distance = np.sqrt(np.sum(halo_pos.astype(np.float64)**2, axis=1))
    within_distance = ((halo_distance < (max_particle_distance + max_radius)) &
                       (halo_distance > (min_particle_distance - max_radius)))
    del halo_distance
    halo_pos = halo_pos[within_distance,:]
    halo_id = halo_id[within_distance]
    halo_radius = halo_radius[within_distance]
    halo_mass = halo_mass[within_distance]
    nr_halos_left = comm.allreduce(halo_id.shape[0], op=MPI.SUM)
    message(f"Halos within batch distance range = {nr_halos_left} of {nr_halos_total}")

    # Determine the range of x coordinates of halos which could overlap
    # particles on this rank (empty ranks get an empty range via +/- inf)
    local_x_min = (float(np.amin(part_pos[:,0])) - max_radius) if nr_particles > 0 else np.inf
    local_x_max = (float(np.amax(part_pos[:,0])) + max_radius) if nr_particles > 0 else -np.inf
    x_min_on_rank = np.asarray(comm.allgather(local_x_min), dtype=np.float64)
    x_max_on_rank = np.asarray(comm.allgather(local_x_max), dtype=np.float64)

    # Sort local halos by x coordinate
    message("Sorting local lightcone halos by x coordinate")
    order = np.argsort(halo_pos[:,0])
    halo_pos = halo_pos[order,:]
    halo_id = halo_id[order]
    halo_radius = halo_radius[order]
    halo_mass = halo_mass[order]
    del order

    # Determine what range of halos needs to be sent to each MPI rank
    first_halo_for_rank = np.searchsorted(halo_pos[:,0], x_min_on_rank, side="left")
    last_halo_for_rank = np.searchsorted(halo_pos[:,0], x_max_on_rank, side="right")
    nr_halos_for_rank = last_halo_for_rank - first_halo_for_rank
    nr_halos_for_rank_total = comm.allreduce(nr_halos_for_rank)
    total_nr_halos_read = comm.allreduce(halo_pos.shape[0])
    total_nr_halos_sent = np.sum(nr_halos_for_rank_total)
    duplication_factor = total_nr_halos_sent / max(total_nr_halos_read, 1)
    message(f"Halos on rank after exchange: min={np.amin(nr_halos_for_rank_total)}, "
            f"max={np.amax(nr_halos_for_rank_total)}, duplication={duplication_factor:.2f}")

    # Compute lengths and offsets for alltoallv halo exchange
    send_offset = first_halo_for_rank
    send_count = nr_halos_for_rank
    recv_count = np.asarray(comm.alltoall(send_count), dtype=send_count.dtype)
    recv_offset = np.cumsum(recv_count) - recv_count

    # Exchange halo IDs
    halo_id_recv = np.empty_like(halo_id, shape=np.sum(recv_count))
    psort.my_alltoallv(halo_id, send_count, send_offset,
                       halo_id_recv, recv_count, recv_offset, comm=comm)
    halo_id = halo_id_recv
    del halo_id_recv

    # Exchange halo radii
    halo_radius_recv = np.empty_like(halo_radius, shape=np.sum(recv_count))
    psort.my_alltoallv(halo_radius, send_count, send_offset,
                       halo_radius_recv, recv_count, recv_offset, comm=comm)
    halo_radius = halo_radius_recv
    del halo_radius_recv

    # Exchange halo masses
    halo_mass_recv = np.empty_like(halo_mass, shape=np.sum(recv_count))
    psort.my_alltoallv(halo_mass, send_count, send_offset,
                       halo_mass_recv, recv_count, recv_offset, comm=comm)
    halo_mass = halo_mass_recv
    del halo_mass_recv

    # Exchange halo positions (flatten, exchange, restore shape)
    halo_pos = np.ascontiguousarray(halo_pos)
    halo_pos.shape = (-1,)
    halo_pos_recv = np.empty_like(halo_pos, shape=3*np.sum(recv_count))
    psort.my_alltoallv(halo_pos, send_count*3, send_offset*3,
                       halo_pos_recv, recv_count*3, recv_offset*3, comm=comm)
    halo_pos = halo_pos_recv
    halo_pos.shape = (-1, 3)
    del halo_pos_recv
    gc.collect()
    comm.barrier()

    # Diagnostics: how often do ranks get zero halos after exchange?
    local_nhalos = int(halo_id.shape[0])
    n_empty_ranks = comm.allreduce(1 if local_nhalos == 0 else 0, op=MPI.SUM)
    if n_empty_ranks > 0:
        message(f"Halo-exchange result: {n_empty_ranks}/{comm_size} ranks have zero halos")

    skip_computation = (local_nhalos == 0)

    # Allocate output arrays for the particle halo IDs etc.
    # NOTE: no full-size per-particle mass array unless the method needs one.
    part_halo_id = -np.ones(nr_particles, dtype=np.int64)
    part_halo_r_frac_2 = np.empty(nr_particles, dtype=np.float32)
    part_halo_r_frac_2[:] = np.finfo(np.float32).max
    if overlap_method == MOST_MASSIVE:
        part_halo_mass = np.full(nr_particles, -1.0, dtype=np.float32)
    elif overlap_method == LEAST_MASSIVE:
        part_halo_mass = np.full(nr_particles, np.finfo(np.float32).max, dtype=np.float32)
    else:
        part_halo_mass = None
    if overlap_method == MASS_WEIGHTED:
        # Criterion value r^2 / M_halo: smaller wins
        part_halo_metric = np.full(nr_particles, np.finfo(np.float32).max, dtype=np.float32)
    else:
        part_halo_metric = None

    nr_assigned = 0
    if not skip_computation:

        # Sort received halos by x so we can window them per particle chunk
        order = np.argsort(halo_pos[:,0])
        halo_pos = halo_pos[order,:]
        halo_id = halo_id[order]
        halo_radius = halo_radius[order]
        halo_mass = halo_mass[order]
        del order
        halo_x = halo_pos[:,0]
        local_max_halo_radius = float(np.amax(halo_radius))

        message("Assigning halo IDs to particles (chunked trees)")
        # part_pos is locally sorted by x after the parallel sort, so chunks
        # are contiguous slabs in x and we can window the halo list per chunk.
        for c0 in range(0, nr_particles, tree_chunk_size):
            c1 = min(c0 + tree_chunk_size, nr_particles)
            chunk_pos = part_pos[c0:c1]

            # Halos whose search sphere can reach this chunk's x-slab
            xlo = chunk_pos[0,0]  - local_max_halo_radius
            xhi = chunk_pos[-1,0] + local_max_halo_radius
            hj0 = np.searchsorted(halo_x, xlo, side="left")
            hj1 = np.searchsorted(halo_x, xhi, side="right")
            if hj1 <= hj0:
                continue

            tree = scipy.spatial.cKDTree(chunk_pos)

            # Query halos in batches with per-halo radii, threaded
            for j0 in range(hj0, hj1, halo_query_batch):
                j1 = min(j0 + halo_query_batch, hj1)
                idx_lists = tree.query_ball_point(halo_pos[j0:j1,:],
                                                  halo_radius[j0:j1],
                                                  workers=-1)
                for k in range(j1 - j0):
                    i = j0 + k
                    if len(idx_lists[k]) == 0:
                        continue
                    idx = np.asarray(idx_lists[k], dtype=np.int64)
                    r_part_2 = np.sum((chunk_pos[idx,:] - halo_pos[i,:])**2, axis=1)
                    r_frac_2 = r_part_2 / (halo_radius[i]**2)
                    gidx = idx + c0

                    if overlap_method == FRACTIONAL_RADIUS:
                        # Halo with the smallest r/R_halo wins
                        to_update = (r_frac_2 < part_halo_r_frac_2[gidx])
                    elif overlap_method == MOST_MASSIVE:
                        to_update = (halo_mass[i] > part_halo_mass[gidx])
                    elif overlap_method == LEAST_MASSIVE:
                        to_update = (halo_mass[i] < part_halo_mass[gidx])
                    elif overlap_method == MASS_WEIGHTED:
                        # Halo minimising r^2/M wins (fixed: previous version
                        # cancelled the radius term out of both sides)
                        metric = (r_part_2 / halo_mass[i]).astype(np.float32)
                        to_update = (metric < part_halo_metric[gidx])
                    else:
                        raise ValueError("Unrecognized value of overlap_method")

                    sel = gidx[to_update]
                    part_halo_id[sel] = halo_id[i]
                    part_halo_r_frac_2[sel] = r_frac_2[to_update]
                    if part_halo_mass is not None:
                        part_halo_mass[sel] = halo_mass[i]
                    if part_halo_metric is not None:
                        part_halo_metric[sel] = metric[to_update]
                    nr_assigned += sel.size

            del tree
            del chunk_pos
        gc.collect()

    nr_assigned_tot = comm.allreduce(nr_assigned)
    fraction_assigned = nr_assigned_tot / nr_particles_total
    message(f"Total particles assigned to halos = {nr_assigned_tot}")
    message(f"Fraction assigned = {fraction_assigned} (inc. duplicates due to halo overlap)")

    # Tidy up
    del halo_id, halo_pos, halo_radius, halo_mass, part_pos
    if part_halo_mass is not None:
        del part_halo_mass
    if part_halo_metric is not None:
        del part_halo_metric
    gc.collect()

    # r_frac = r/R for particles in halos, -1 for those not in halos
    in_halo = (part_halo_id >= 0)
    part_halo_r_frac = np.where(in_halo, np.sqrt(part_halo_r_frac_2),
                                np.float32(-1.0)).astype(np.float32)
    del part_halo_r_frac_2, in_halo

    # Restore original particle ordering: direct scatter using the known
    # entry partitioning instead of a second full parallel sort.
    message("Restoring particle order")
    part_halo_id, part_halo_r_frac = restore_particle_order(
        part_index, [part_halo_id, part_halo_r_frac], entry_counts)
    del part_index

    return part_halo_id, part_halo_r_frac


def main(args):

    # Determine method to deal with overlapping halos
    overlap_method = overlap_methods[args.overlap_method]
    message(f"Halo overlap method: {args.overlap_method}")

    # Radius/mass definitions (BoundSubhalo so satellites are included)
    radius_name = "BoundSubhalo/EncloseRadius"
    mass_name   = "BoundSubhalo/TotalMass"
    message(f"Halo radius definition: {radius_name}")

    # Read in position and radius for halos in the lightcone
    halo_lightcone_data = read_lightcone_halo_positions_and_radii(args, radius_name, mass_name)

    # Locate the particle data, grouped into batches by shell/redshift
    type_z_range, file_batches = read_lightcone_index(args)

    halo_id_all = halo_lightcone_data["IndexInHaloLightcone"]
    halo_pos_all = halo_lightcone_data["Lightcone/HaloCentre"]
    halo_radius_all = halo_lightcone_data[radius_name]
    halo_mass_all = halo_lightcone_data[mass_name]

    # ------------------------------------------------------------------
    # Pass 1: assign particles to halos, one batch of files at a time
    # ------------------------------------------------------------------
    for batch_nr, batch_files in enumerate(file_batches):

        message(f"=== Pass 1: batch {batch_nr+1} of {len(file_batches)} "
                f"({len(batch_files)} files) ===")

        # Output filenames for this batch: input names with directory replaced
        batch_outputs = [os.path.join(args.output_dir, os.path.split(f)[1])
                         for f in batch_files]

        # Open the input particle file set for this batch
        mf = phdf5.MultiFile(batch_files, comm=comm)

        create_files = True
        for ptype in type_z_range:

            message(f"Processing particle type {ptype}")

            # Read positions and cast to float32 (halves the dominant array)
            part_pos = mf.read("Coordinates", group=ptype)
            part_pos = np.ascontiguousarray(part_pos, dtype=np.float32)

            # Record number of particles read from each file
            elements_per_file = mf.get_elements_per_file("Coordinates", group=ptype)

            # Rebalance particle load between MPI ranks
            nr_parts_per_rank_read = np.asarray(comm.allgather(part_pos.shape[0]), dtype=int)
            nr_parts_total = np.sum(nr_parts_per_rank_read)
            nr_parts_per_rank_balanced = np.zeros_like(nr_parts_per_rank_read)
            nr_av = (nr_parts_total // comm_size)
            nr_parts_per_rank_balanced[:] = nr_av
            nr_parts_per_rank_balanced[:nr_parts_total % comm_size] += 1
            assert np.sum(nr_parts_per_rank_balanced) == nr_parts_total
            part_pos = psort.repartition(part_pos, ndesired=nr_parts_per_rank_balanced, comm=comm)

            # Assign group indexes to the particles
            part_halo_id, part_halo_r_frac = compute_particle_group_index(
                halo_id_all, halo_pos_all, halo_radius_all, halo_mass_all,
                part_pos, overlap_method,
                tree_chunk_size=args.tree_chunk_size,
                halo_query_batch=args.halo_query_batch)
            del part_pos

            # Restore original partitioning of particles
            part_halo_id = psort.repartition(part_halo_id, ndesired=nr_parts_per_rank_read, comm=comm)
            part_halo_r_frac = psort.repartition(part_halo_r_frac, ndesired=nr_parts_per_rank_read, comm=comm)

            # Write the output, appending to file if not the first type.
            # NOTE: no per-particle HaloMass here -- pass 2 writes TotalMass
            # (and other halo properties) via IndexInHaloLightcone.
            message(f"Writing output to {args.output_dir}")
            mode = "w" if create_files else "r+"
            datasets = {
                "IndexInHaloLightcone" : part_halo_id,
                "FractionalRadius"     : part_halo_r_frac,
            }
            attributes = {
                "IndexInHaloLightcone" : halo_lightcone_data["InputHalos/HaloCatalogueIndex"].attrs,
                "FractionalRadius"     : halo_lightcone_data["InputHalos/HaloCatalogueIndex"].attrs,
            }
            mf.write(datasets, elements_per_file, batch_outputs, mode,
                     group=ptype, attrs=attributes, gzip=6, shuffle=True)

            del part_halo_id
            del part_halo_r_frac
            gc.collect()

            create_files = False

        del mf
        gc.collect()
        report_memory(f"end of pass-1 batch {batch_nr+1}")

    comm.barrier()

    # Discard reordered halo lightcone data
    del halo_id_all, halo_pos_all, halo_radius_all, halo_mass_all
    del halo_lightcone_data
    gc.collect()

    # ------------------------------------------------------------------
    # Pass 2: copy halo properties onto the particles, batched as above.
    #
    # Consider trimming this list: anything joinable later via SOAPIndex /
    # HaloCatalogueIndex can be dropped to save memory and disk (HaloCentre
    # alone is 24 bytes per particle).
    # ------------------------------------------------------------------
    message("Reading lightcone halo properties to copy to output particle files")
    halo_properties = (
        "BoundSubhalo/TotalMass",
        "Lightcone/HaloCentre",
        "Lightcone/Redshift",
        "Lightcone/SnapshotNumber",
        "InputHalos/HaloCatalogueIndex",
        "InputHalos/SOAPIndex",
    )
    mf_in = phdf5.MultiFile(args.halo_lightcone_filenames,
                            file_idx=np.arange(args.first_snap_nr, args.final_snap_nr+1),
                            comm=comm)
    halo_lightcone_data = mf_in.read(halo_properties, read_attributes=True)
    halo_lightcone_data["InputHalos/HaloCatalogueIndex"] = \
        halo_lightcone_data["InputHalos/HaloCatalogueIndex"].astype(np.int64)
    halo_lightcone_data["InputHalos/SOAPIndex"] = \
        halo_lightcone_data["InputHalos/SOAPIndex"].astype(np.int64)

    for batch_nr, batch_files in enumerate(file_batches):

        message(f"=== Pass 2: batch {batch_nr+1} of {len(file_batches)} ===")
        batch_outputs = [os.path.join(args.output_dir, os.path.split(f)[1])
                         for f in batch_files]

        # Open the set of particle files to update for this batch
        mf_out = phdf5.MultiFile(batch_outputs, comm=comm)

        for ptype in type_z_range:

            message(f"Reading halo index for particles of type {ptype}")
            halo_index = mf_out.read(f"{ptype}/IndexInHaloLightcone")
            elements_per_file = mf_out.get_elements_per_file(f"{ptype}/IndexInHaloLightcone")
            in_halo = (halo_index >= 0)

            for prop_name in halo_properties:
                message(f"Pass through quantity {prop_name} for type {ptype}")
                dtype = halo_lightcone_data[prop_name].dtype
                shape = (halo_index.shape[0],) + halo_lightcone_data[prop_name].shape[1:]
                prop_data = -np.ones(shape, dtype=dtype) # property=-1 if not in halo
                prop_data[in_halo,...] = psort.fetch_elements(
                    halo_lightcone_data[prop_name], halo_index[in_halo], comm=comm)
                dataset_name = f"{ptype}/{prop_name.split('/')[-1]}"
                mf_out.write({dataset_name : prop_data}, elements_per_file, batch_outputs, "r+",
                             attrs={dataset_name : halo_lightcone_data[prop_name].attrs},
                             gzip=6, shuffle=True)
                del prop_data
                gc.collect()

            del halo_index, in_halo
            gc.collect()

        del mf_out
        gc.collect()
        report_memory(f"end of pass-2 batch {batch_nr+1}")

    del mf_in, halo_lightcone_data


if __name__ == "__main__":

    # Get command line arguments
    from virgo.mpi.util import MPIArgumentParser
    parser = MPIArgumentParser(description='Create lightcone halo catalogues.', comm=comm)
    parser.add_argument('lightcone_dir',  help='Directory with lightcone particle outputs')
    parser.add_argument('lightcone_base', help='Base name of the lightcone to use')
    parser.add_argument('halo_lightcone_filenames', help='Format string to generate halo lightcone filenames')
    parser.add_argument('soap_filenames', help='Format string to generate SOAP filenames')
    parser.add_argument('output_dir',     help='Where to write the output')
    parser.add_argument('--overlap-method', type=str, default="fractional-radius",
                        choices=list(overlap_methods),
                        help="How to assign particles which are in overlapping halos")
    parser.add_argument('--files-per-batch', type=int, default=64,
                        help="Approx. number of particle files to process per batch "
                             "(main memory knob: smaller = less memory)")
    parser.add_argument('--tree-chunk-size', type=int, default=5_000_000,
                        help="Max particles per KDTree built on each rank")
    parser.add_argument('--halo-query-batch', type=int, default=4096,
                        help="Number of halos per vectorised tree query")
    parser.add_argument('--radius-multiplier', type=float, default=2.0,
                        help="Multiply halo search radii by this factor (was hardcoded 2x)")
    parser.add_argument('--first-snap-nr', type=int, default=72,
                        help="First snapshot in the halo lightcone files (z=0.2 for L1000N1800)")
    parser.add_argument('--final-snap-nr', type=int, default=77,
                        help="Final snapshot in the halo lightcone files (z=0 for L1000N1800)")
    args = parser.parse_args()

    message(f"Starting on {comm_size} MPI ranks")
    main(args)
    message("Done.")