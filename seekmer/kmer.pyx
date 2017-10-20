# cython: language_level=3

'''K-mer and related classes'''

import contextlib
import gzip
import os
import re
import tempfile

import logbook
import numpy
import tables

cimport libc.stdlib
cimport cython
cimport numpy


numpy.import_array()

_log = logbook.Logger('seekmer.kmer')
cdef int _KMER_LENGTH = 31
_KMER_PATTERN = re.compile(f'^[ACGTacgt]{{{_KMER_LENGTH}}}$')
cdef unsigned long long int _INVALID_KMER = 0xFFFFFFFFFFFFFFFF
cdef unsigned long long int _KMER_MASK = ~(_INVALID_KMER << (2 * _KMER_LENGTH))
cdef int _KMER_HASH_SIZE = 0x10000000
cdef int _CONTIG_SIZE = 0x10000000
cdef int _BUFFER_SIZE = 0x1000
cdef int _KMER_HASH_MASK = _KMER_HASH_SIZE - 1
cdef unsigned int _FULL_MASK = 0xFFFFFFFF
cdef const char *_BASES = b'ACGT'
cdef unsigned long long int *_MASKS = [
    0x3333333333333333,
    0x0F0F0F0F0F0F0F0F,
    0x00FF00FF00FF00FF,
    0x0000FFFF0000FFFF,
    0x00000000FFFFFFFF,
]


@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.wraparound(False)
cdef unsigned long long int _encode_kmer(char [:] sequence) nogil:
    '''Encodes a K-mer sequence into a binary representation.

    It assumes the length and all the characters of K-mer are valid.

    Parameters
    ----------
    sequence : 1D char MemoryView
        The sequence bytestring. It must be of the same length of the
        predefined K-mer length, and only contains 'ACGTacgt'.

    Returns
    -------
    kmer : unsigned long long int
        The encoded K-mer.
    '''
    cdef unsigned long long int kmer = 0
    for i in range(_KMER_LENGTH):
        if sequence[i] == 65:
            kmer = kmer << 2
        elif sequence[i] == 67:
            kmer = (kmer << 2) | 1
        elif sequence[i] == 71:
            kmer = (kmer << 2) | 2
        elif sequence[i] == 84:
            kmer = (kmer << 2) | 3
    return kmer


cpdef unsigned long long int encode_kmer(str sequence):
    '''Encodes a K-mer sequence into a binary representation.

    Parameters
    ----------
    sequence : str
        The sequence string. It must be of the same length of the
        predefined K-mer length and only contains 'ACGTacgt'.

    Returns
    -------
    kmer : unsigned long long int
        The encoded K-mer.
    '''
    if not _KMER_PATTERN.match(sequence):
        raise ValueError(f'invalid K-mer sequence', sequence)
    return _encode_kmer(sequence)


cpdef str decode_kmer(unsigned long long int kmer):
    '''Decodes a K-mer binary representation back to a bytestring.

    Parameters
    ----------
    kmer : unsigned long long int
        The encoded K-mer.

    Returns
    -------
    seq : bytes
        The sequence bytestring.
    '''
    cdef char *cseq = <char *>libc.stdlib.malloc(
        (_KMER_LENGTH + 1) * sizeof(char)
    )
    cdef int i = 0
    with nogil:
        cseq[_KMER_LENGTH] = 0
        for i in range(_KMER_LENGTH):
            cseq[i] = _BASES[(kmer >> (2 * (_KMER_LENGTH - 1 - i))) & 3]
    cdef bytes seq
    try:
        seq = cseq
    finally:
        libc.stdlib.free(cseq)
    return seq.decode()


cdef int hash_kmer(unsigned long long int kmer) nogil:
    '''Hashes a K-mer using DJB algorithm.

    Parameters
    ----------
    kmer : unsigned long long int
        The encoded K-mer.

    Returns
    -------
    h : unsigned int
        The hashed value.
    '''
    cdef unsigned int h = 5381
    h = ((h << 5) + h) ^ <unsigned int>(kmer >> 32)
    h = ((h << 5) + h) ^ <unsigned int>(kmer)
    return <int>(h & _KMER_HASH_MASK)


cdef unsigned long long int reverse_complement_kmer(
    unsigned long long int kmer
) nogil:
    '''Returns the reverse-complemented of a K-mer sequence.

    Parameters
    ----------
    kmer : unsigned long long int
        The orignal K-mer.

    Returns
    -------
    v : unsigned long long int
        The reversed K-mer.
    '''
    kmer = ~kmer
    cdef int shift = 1
    for i in range(5):
        shift <<= 1
        kmer = ((kmer & _MASKS[i]) << shift) | ((kmer >> shift) & _MASKS[i])
    kmer >>= 2 * (32 - _KMER_LENGTH)
    return kmer


@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.wraparound(False)
cdef int hash_find_kmer(unsigned long long int [:] kmers,
                        unsigned long long int kmer) nogil:
    '''Returns the index of the given K-mer in the hash table.

    It returns -1 if the K-mer is not registered.

    Parameters
    ----------
    kmers : 1-D MemoryView of unsigned long long int
        The K-mer hash table.
    kmer : unsigned long long int
        A K-mer.

    Returns
    -------
    index : int
        The index of the given K-mer in the hash table, or -1 if the
        K-mer is not in the hash table.
    '''
    cdef int offset = hash_kmer(kmer)
    if kmers[offset] == _INVALID_KMER:
        return -1
    if kmers[offset] == kmer:
        return offset
    cdef int i = (offset + 1) & _KMER_HASH_MASK
    while i != offset:
        if kmers[i] == _INVALID_KMER:
            return -1
        if kmers[i] == kmer:
            return i
        i += 1
        i &= _KMER_HASH_MASK
    return -1


@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.wraparound(False)
cdef int hash_find_kmer_insert(unsigned long long int [:] kmer_hash,
                               unsigned long long int kmer) nogil:
    '''Returns the index of the given K-mer in the hash table.

    It returns the index of the next empty slot for insertion if the
    K-mer is not registered, or -1 if the K-mer is not registered and
    there is no empty slots in the hash table.

    Parameters
    ----------
    kmer_hash : 1-D MemoryView of unsigned long long int
        The K-mer hash table.
    kmer : unsigned long long int
        A K-mer.

    Returns
    -------
    index : int
        The index of where the given K-mer should be placed in the hash
        table, or -1 if the K-mer cannot be added to the hash table.
    '''
    cdef int offset = hash_kmer(kmer)
    if kmer_hash[offset] == _INVALID_KMER or kmer_hash[offset] == kmer:
        return offset
    cdef int i = (offset + 1) & _KMER_HASH_MASK
    while i != offset:
        if kmer_hash[i] == _INVALID_KMER or kmer_hash[i] == kmer:
            return i
        i += 1
        i &= _KMER_HASH_MASK
    return -1


@contextlib.contextmanager
def create_temporary_hdf5():
    with tempfile.NamedTemporaryFile('wb', delete=False) as f:
        tmp_name = f.name
    filters = tables.Filters(complevel=1, complib='blosc')
    tmp = tables.open_file(tmp_name, mode='w', filters=filters)
    tmp.close()
    tmp = tables.open_file(tmp_name, mode='r+', filters=filters)
    try:
        yield tmp
    except:
        raise
    finally:
        tmp.close()
        os.remove(tmp_name)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef object _initialise_kmer_hash():
    kmers = numpy.recarray(
        _KMER_HASH_SIZE,
        dtype=[('kmer', 'u8'), ('prepend', 'i4'), ('append', 'i4')],
    )
    kmers['kmer'][:] = _INVALID_KMER
    kmers['prepend'][:] = -1
    kmers['append'][:] = -1
    return kmers


cdef struct kmer_hash_view:
    unsigned long long int [:] kmer
    int [:] prepend
    int [:] append


@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.wraparound(False)
cdef kmer_hash_view _view_kmer_hash(object kmers):
    cdef kmer_hash_view kmer_view
    kmer_view.kmer = kmers[kmers.dtype.names[0]]
    kmer_view.prepend = kmers[kmers.dtype.names[1]]
    kmer_view.append = kmers[kmers.dtype.names[2]]
    return kmer_view


@cython.boundscheck(False)
@cython.wraparound(False)
cdef object _initialise_kmer_record():
    records = numpy.recarray(
        _BUFFER_SIZE,
        dtype=[('kmer', 'u8'), ('transcript', 'i4'), ('coordinate', 'i4')],
    )
    records[:] = 0
    return records


cdef struct kmer_record_view:
    unsigned long long int [:] kmer
    int [:] transcript
    int [:] coordinate


@cython.initializedcheck(False)
cdef kmer_record_view _view_kmer_record(object records):
    cdef kmer_record_view rec_view
    rec_view.kmer = records['kmer']
    rec_view.transcript = records['transcript']
    rec_view.coordinate = records['coordinate']
    return rec_view


cdef struct kmer_graph_state:
    unsigned long long int last_kmer
    bint new
    bint forward


@cython.boundscheck(False)
@cython.wraparound(False)
def fetch_fasta_entries(fasta_file):
    '''Fetch names and sequences of all entries in the FASTA file.

    Parameters
    ----------
    fasta_file : file-like object
        A FASTA file.

    Yields
    ------
    name : bytes
        The entry name
    sequence: bytes
        The sequence
    '''
    name = None
    seq = []
    for line in fasta_file:
        line = line.strip()
        if line[0] != ord(b'>'):
            seq.append(line)
            continue
        if name is not None:
            yield name, b''.join(seq)
            seq.clear()
        name = line[1:].split()[0]
    if name is not None:
        yield name, b''.join(seq)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef bint _update_kmer(
    unsigned long long int *kmer, unsigned int *soft_mask,
    unsigned int *hard_mask, int *transcript_len, char base,
) nogil:
    '''Check the validity of the base and appends it to the K-mer.

    If the base is not one of "ACGT", the function returns False.

    Parameters
    ----------
    kmer : pointer to unsigned long long int
        The K-mer to update.
    soft_mask : pointer to unsigned int
        The soft mask bit array.
    hard_mask : pointer to unsigned int
        The hard mask bit array.
    base : char
        The base.

    Returns
    -------
    valid : bool
        Whether the base is valid.
    '''
    kmer[0] <<= 2
    hard_mask[0] <<= 1
    soft_mask[0] <<= 1
    if 96 < base < 123:
        base -= 32
        soft_mask[0] |= 1
    if base == ord(b'A'):
        pass
    elif base == ord(b'C'):
        kmer[0] |= 1
    elif base == ord(b'G'):
        kmer[0] |= 2
    elif base == ord(b'T'):
        kmer[0] |= 3
    else:
        soft_mask[0] |= 1
        hard_mask[0] |= 1
        transcript_len[0] -= 1
        return False
    if soft_mask[0] == _FULL_MASK:
        transcript_len[0] -= 1
        return False
    if hard_mask[0] != 0:
        return False
    return True


@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.wraparound(False)
cdef int _add_record(kmer_record_view record_view, int record_idx,
                     unsigned long long int kmer, int transcript_id,
                     int kmer_coord) nogil:
    '''Appends a K-mer record and returns the index of next slot.'''
    record_view.kmer[record_idx] = kmer
    record_view.transcript[record_idx] = transcript_id
    record_view.coordinate[record_idx] = kmer_coord
    return record_idx + 1


@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.wraparound(False)
cdef void _disconnect_kmer(
    kmer_hash_view kmer_view, int idx, bint forward,
) nogil:
    '''Disconnect a K-mer on one direction in the De Bruijn graph.'''
    cdef int linked_idx
    if forward:
        linked_idx = kmer_view.append[idx]
        kmer_view.append[idx] = -1
    else:
        linked_idx = kmer_view.prepend[idx]
        kmer_view.prepend[idx] = -1
    if linked_idx != -1:
        if kmer_view.append[linked_idx] == idx:
            kmer_view.append[linked_idx] = -1
        else:
            kmer_view.prepend[linked_idx] = -1


@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.wraparound(False)
cdef kmer_graph_state _register_kmer(kmer_hash_view kmer_view,
                                     unsigned long long int kmer,
                                     kmer_graph_state state) nogil:
    '''Adds K-mer into the hash table.

    It also creates a connection map of the De Bruijn graph, which will
    be used later for updating contig information.

    The function needs to update related information for registering
    upcoming K-mers.
    '''
    cdef unsigned long long int rc_kmer = reverse_complement_kmer(kmer)
    cdef bint forward = kmer < rc_kmer
    if not forward:
        kmer = rc_kmer
    cdef int kmer_idx = hash_find_kmer_insert(kmer_view.kmer, kmer)
    if state.last_kmer == _INVALID_KMER:
        if kmer_view.kmer[kmer_idx] != kmer:
            kmer_view.kmer[kmer_idx] = kmer
            state.new = True
        else:
            _disconnect_kmer(kmer_view, kmer_idx, not forward)
            state.new = False
        state.forward = forward
        state.last_kmer = kmer
        return state
    cdef int last_idx = hash_find_kmer(kmer_view.kmer, state.last_kmer)
    if kmer_view.kmer[kmer_idx] != kmer:
        kmer_view.kmer[kmer_idx] = kmer
        if not state.new:
            _disconnect_kmer(kmer_view, last_idx, state.forward)
        else:
            if state.forward:
                kmer_view.append[last_idx] = kmer_idx
            else:
                kmer_view.prepend[last_idx] = kmer_idx
            if forward:
                kmer_view.prepend[kmer_idx] = last_idx
            else:
                kmer_view.append[kmer_idx] = last_idx
        state.new = True
        state.forward = forward
        state.last_kmer = kmer
        return state
    if (((not state.forward and kmer_view.prepend[last_idx] == kmer_idx) or
             (state.forward and kmer_view.append[last_idx] == kmer_idx)) and
            ((not forward and kmer_view.append[kmer_idx] == last_idx) or
             (forward and kmer_view.prepend[kmer_idx] == last_idx))):
        state.new = False
        state.forward = forward
        state.last_kmer = kmer
        return state
    _disconnect_kmer(kmer_view, last_idx, state.forward)
    _disconnect_kmer(kmer_view, kmer_idx, not forward)
    if ((forward and kmer_view.append[kmer_idx] == -1) or
            (not forward and kmer_view.prepend[kmer_idx] == -1)):
        state.new = True
        state.forward = True
        state.last_kmer = _INVALID_KMER
        return state
    state.new = False
    state.forward = forward
    state.last_kmer = kmer
    return state


@cython.initializedcheck(False)
cdef kmer_graph_state _terminate_contig(kmer_hash_view kmer_view,
                                        kmer_graph_state state) nogil:
    '''Terminate the continuation of the contig.'''
    cdef int idx
    if state.last_kmer != _INVALID_KMER:
        idx = hash_find_kmer(kmer_view.kmer, state.last_kmer)
        _disconnect_kmer(kmer_view, idx, state.forward)
    state.new = True
    state.forward = True
    state.last_kmer = _INVALID_KMER
    return state


@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.wraparound(False)
cdef object _collect_kmer(object fasta_file, object tmp_file):
    '''Collect all K-mers and transcripts from FASTA file.

    It collects all K-mers and stores them in a hash table. The table
    collects the contig information, too. The coordinate of all K-mers
    against the reference is stored in a temporary file, together with
    transcript information.

    Parameters
    ----------
    fasta_file: file-like object
        A FASTA file.
    tmp_path: tables.File
        A temporary HDF5 file for coordinate information

    Returns
    -------
    kmers: numpy.recarray
        A K-mer hash table with their contig information.
    '''
    kmers = _initialise_kmer_hash()
    cdef kmer_hash_view kmer_view = _view_kmer_hash(kmers)
    cdef int kmer_count = 0
    transcripts = []
    cdef int trx_idx = 0
    rec = _initialise_kmer_record()
    cdef kmer_record_view rec_view = _view_kmer_record(rec)
    rec_table = tmp_file.create_table('/', 'rec', description=rec.dtype,
                                      expectedrows=2 * _KMER_HASH_SIZE)
    cdef int rec_idx = 0
    cdef unsigned long long int kmer = _INVALID_KMER
    cdef unsigned int hard_mask = _FULL_MASK
    cdef unsigned int soft_mask = _FULL_MASK
    cdef kmer_graph_state state
    cdef int trx_len = 0
    cdef char *cseq = NULL
    cdef char base
    cdef int kmer_coord
    max_trx_name_len = 0
    for trx_idx, (name, seq) in enumerate(fetch_fasta_entries(fasta_file)):
        trx_len = len(seq)
        cseq = seq
        kmer = _INVALID_KMER
        hard_mask = _FULL_MASK
        soft_mask = _FULL_MASK
        state.new = True
        state.forward = True
        state.last_kmer = _INVALID_KMER
        for kmer_coord, base in enumerate(cseq):
            if not _update_kmer(&kmer, &soft_mask, &hard_mask, &trx_len, base):
                state = _terminate_contig(kmer_view, state)
                continue
            kmer &= _KMER_MASK
            rec_idx = _add_record(rec_view, rec_idx, kmer, trx_idx, kmer_coord)
            if rec_idx >= _BUFFER_SIZE:
                rec_table.append(rec)
                rec[:] = 0
                rec_idx = 0
            state = _register_kmer(kmer_view, kmer, state)
            kmer_count += 1
        state = _terminate_contig(kmer_view, state)
        name = name.split()[0]
        if len(name) > max_trx_name_len:
            max_trx_name_len = len(name)
        transcripts.append((name, len(seq), trx_len))
    rec_table.append(rec[:rec_idx])
    cdef int uniq_kmer_count = numpy.count_nonzero(kmers.kmer != _INVALID_KMER)
    _log.info(f'Hashed {uniq_kmer_count} unique K-mers.')
    _log.info(f'Collected {kmer_count} K-mer mapping entries.')
    transcripts = numpy.rec.fromrecords(
        transcripts,
        dtype=[('name', f'S{max_trx_name_len}'), ('length', 'i4'),
               ('effective_length', 'i4')],
    )
    _log.info(f'Indexed {transcripts.size} transcripts.')
    return kmers, transcripts


cdef int _hash_to_map(int idx) nogil:
    '''Map new IDs to (-inf, -2] for temporary storage.'''
    return -2 - idx


cdef struct kmer_map_view:
    unsigned long long int [:] kmer
    int [:] contig
    int [:] offset


@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.wraparound(False)
cdef kmer_map_view _view_kmer_map(object kmers):
    cdef kmer_map_view kmer_view
    kmer_view.kmer = kmers[kmers.dtype.names[0]]
    kmer_view.contig = kmers[kmers.dtype.names[1]]
    kmer_view.offset = kmers[kmers.dtype.names[2]]
    return kmer_view


@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.wraparound(False)
cdef void _swap_kmer(kmer_hash_view kmer_view, int i):
    kmer_view.kmer[i] = reverse_complement_kmer(kmer_view.kmer[i])
    cdef int tmp = kmer_view.prepend[i]
    kmer_view.prepend[i] = kmer_view.append[i]
    kmer_view.append[i] = tmp


@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.wraparound(False)
cdef void _store_kmer_to_seq(char [:] seq, unsigned long long int kmer,
                             int offset) nogil:
    '''Save a K-mer to a contig bytestring.

    Parameters
    ----------
    seq : 1-D MemoryView of char
        The output contig sequences.
    kmer : unsigned long long int
        The encoded K-mer.
    offset: int
        The start writing position of the original sequence.
    '''
    cdef int i
    for i in range(_KMER_LENGTH):
        seq[i + offset] = _BASES[(kmer >> (2 * (_KMER_LENGTH - 1 - i))) & 3]


@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.wraparound(False)
cdef object _assemble_contigs(object kmers, object tmp_file):
    '''Maps K-mers to contigs and contigs to transcripts.

    Parameters
    ----------
    kmers : numpy.recarray
        A K-mer hash table.
    '''
    cdef kmer_hash_view hash_view = _view_kmer_hash(kmers)
    cdef kmer_map_view map_view = _view_kmer_map(kmers)
    contig_seq = numpy.zeros(_CONTIG_SIZE, dtype='S1')
    cdef char [:] seq_view = contig_seq
    contigs = []
    cdef int contig_idx = 0
    cdef int contig_start = 0
    cdef int offset = 0
    cdef int i
    cdef unsigned long long int kmer
    cdef int last_idx = -1
    cdef int cur_idx = 0
    for i in range(_KMER_HASH_SIZE):
        if hash_view.kmer[i] == _INVALID_KMER:
            continue
        if hash_view.prepend[i] > -1 and hash_view.append[i] > -1:
            continue
        if hash_view.prepend[i] < -1 and hash_view.append[i] < -1:
            continue
        if hash_view.prepend[i] == -1 and hash_view.append[i] == -1:
            map_view.contig[i] = _hash_to_map(contig_idx)
            map_view.offset[i] = 0
            kmer = map_view.kmer[i]
            _store_kmer_to_seq(seq_view, kmer, contig_start)
            contigs.append((contig_start, _KMER_LENGTH))
            contig_start += _KMER_LENGTH
            contig_idx += 1
            continue
        offset = contig_start
        last_idx = -1
        cur_idx = i
        while cur_idx != -1:
            if hash_view.append[cur_idx] == last_idx:
                _swap_kmer(hash_view, cur_idx)
            last_idx = cur_idx
            cur_idx = hash_view.append[cur_idx]
            kmer = map_view.kmer[last_idx]
            seq_view[offset] = _BASES[(kmer >> (2 * (_KMER_LENGTH - 1))) & 3]
            map_view.contig[last_idx] = _hash_to_map(contig_idx)
            map_view.offset[last_idx] = _hash_to_map(offset - contig_start)
            offset += 1
        _store_kmer_to_seq(seq_view, kmer, offset - 1)
        offset += _KMER_LENGTH - 1
        contigs.append((contig_start, offset - contig_start))
        contig_start = offset
        contig_idx += 1
    kmers.dtype.names = 'kmer', 'contig', 'offset'
    kmers['contig'] = -2 - kmers['contig']
    kmers['offset'] = -2 - kmers['offset']
    seq_view = None
    contig_seq.resize(contig_start)
    contigs = numpy.rec.fromrecords(
        contigs,
        dtype=[('offset', 'i4'), ('length', 'i4')],
    )
    _log.info(f'Assembled {len(contigs)} contigs, '
              f'containing {contig_seq.size} bases.')
    return kmers, contig_seq, contigs


cdef struct read:
    int length
    char *seq


cdef struct chunk:
    int size
    read *reads


cdef struct read_map:
    int contig
    int offset
    int target_count
    int *targets


cdef struct chunk_map:
    int size
    read_map **reads


cdef class KMerIndex:
    '''Provides an index of K-mers, contigs, and all transcripts.

    Attributes
    ----------
    kmers: numpy.recarray
        All K-mers and their contig mapping.
    contigs: numpy.recarray
        All contigs and their lengths and transcript mapping.
    transcripts: numpy.recarray
        All transcripts and their lengths.
    '''

    cdef object _kmers
    cdef object _contig_seq
    cdef object _contigs
    cdef object _transcripts
    cdef unsigned long long int [:] _kmer_view
    cdef long long int [:] _kmer_contig_map
    cdef long long int [:] _kmer_contig_offset
    cdef long long int [:] _contig_lens
    cdef object [:] _contig_entry_map

    def __init__(self, kmers, contig_seq, contigs, transcripts):
        '''Create a K-mer index.

        Parameters
        ----------
        kmers: numpy.recarray
            All K-mers and their contig mapping.
        contig_seq: numpy.ndarray, dtype='S1'
            All contig sequences.
        contigs: numpy.recarray
            All contigs and their lengths and transcript mapping.
        transcripts: numpy.recarray
            All transcripts and their lengths.
        '''
        self._kmers = kmers
        self._contig_seq = contig_seq
        self._contigs = contigs
        self._transcripts = transcripts

    @property
    def kmers(self):
        return self._kmers

    @kmers.setter
    def kmers(self, v):
        self._kmers = v

    @property
    def contig_seq(self):
        return self._contig_seq

    @contig_seq.setter
    def contig_seq(self, v):
        self._contig_seq = v

    @property
    def contigs(self):
        return self._contigs

    @contigs.setter
    def contigs(self, v):
        self._contigs = v

    @property
    def transcripts(self):
        return self._transcripts

    @transcripts.setter
    def transcripts(self, v):
        self._transcripts = v

    @classmethod
    def from_transcriptome(cls, fasta_path, compress=None):
        '''Builds a mapping index from transcriptome reference.

        Builds a mapping index from transcriptome reference FASTA file.
        ENSEMBL cDNA FASTA file is the most commonly used.

        Parameters
        ----------
        fasta_path: pathlib.Path
            A path to the transcriptome reference FASTA file.
        compression: 'gz' or None, default=None
            The compression format of the reference file. If it is None,
            the format will be chosen based on the suffix of the file.

        Returns
        -------
        index: seekmer.kmer.KMerIndex
            The K-mer mapping index.
        '''
        _log.notice(f'Indexing: {fasta_path}')
        with create_temporary_hdf5() as tmp:
            if fasta_path.suffix == '.gz' or compress == 'gz':
                with gzip.open(str(fasta_path), 'rb') as fasta:
                    kmers, transcripts = _collect_kmer(fasta, tmp)
            else:
                with fasta_path.open('rb') as fasta:
                    kmers, transcripts = _collect_kmer(fasta, tmp)
            kmers, contig_seq, contigs = _assemble_contigs(kmers, tmp)
        return KMerIndex(kmers, contig_seq, contigs, transcripts)

    @classmethod
    def from_index(cls, path):
        '''Load the index from a HDF5 file.

        Parameters
        ----------
        path : pathlib.Path
            The path to the input index file.

        Returns
        -------
        index : seekmer.kmer.KMerIndex
            The K-mer mapping index.
        '''
        _log.notice(f'Loading index: {path}')
        filters = tables.Filters(complevel=9, complib='blosc')
        with tables.open_file(str(path), mode='r', filters=filters) as h5:
            idx = KMerIndex(h5.root.kmers.read(), h5.root.contig_seq.read(),
                            h5.root.contigs.read(), h5.root.transcripts.read())
            count = numpy.count_nonzero(idx._kmers['kmer'] != _INVALID_KMER)
            _log.info(f'Loaded {count} K-mers, {idx._contigs.size} contigs, '
                      f'and {idx._transcripts.size} transcripts.')
            return idx

    cdef void _save_kmers_with_low_memory(self, object h5):
        '''Save K-mer hash table with low memory.

        Parameters
        ----------
        h5 : tables.File
            The output index HDF5 file.
        '''
        tab = h5.create_table('/', 'kmers', description=self._kmers.dtype,
                              expectedrows=_KMER_HASH_SIZE)
        for i in range(0, _KMER_HASH_SIZE, tab.chunkshape[0]):
            tab.append(self._kmers[i:(i + tab.chunkshape[0])])

    def save(self, path):
        '''Save the index to a HDF5 file.

        Parameters
        ----------
        path : pathlib.Path
            The path to the output index file.
        '''
        filters = tables.Filters(complevel=9, complib='blosc')
        with tables.open_file(str(path), mode='w', filters=filters) as h5:
            self._save_kmers_with_low_memory(h5)
            h5.create_carray('/', 'contig_seq', obj=self._contig_seq)
            h5.create_table('/', 'contigs', obj=self._contigs)
            h5.create_table('/', 'transcripts', obj=self._transcripts)

    def align_chunk(self, chunk1, chunk2):
        cdef int i
        cdef object seq
        cdef chunk c1
        c1.size = len(chunk1)
        c1.reads = <read *>libc.stdlib.malloc(c1.size * sizeof(read))
        for i, seq in enumerate(chunk1):
            c1.reads[i].length = len(seq)
            c1.reads[i].seq = seq
        cdef chunk c2
        c2.size = len(chunk1)
        c2.reads = <read *>libc.stdlib.malloc(c2.size * sizeof(read))
        for i, seq in enumerate(chunk2):
            c2.reads[i].length = len(seq)
            c2.reads[i].seq = seq
        cdef chunk_map *result
        with nogil:
            result = self._align_chunk(c1, c2)
        libc.stdlib.free(c1.reads)
        libc.stdlib.free(c2.reads)
        return []

    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    @cython.wraparound(False)
    cdef read_map *_align(self, read read1, read read2) nogil:
        cdef read_map *res = <read_map *>libc.stdlib.malloc(sizeof(read_map))
        res.contig = 0
        res.offset = 0
        res.target_count = 0
        res.targets = NULL
        return res

    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    @cython.wraparound(False)
    cdef chunk_map *_align_chunk(self, chunk chunk1, chunk chunk2) nogil:
        cdef int i = 0
        cdef chunk_map *res = <chunk_map *>libc.stdlib.malloc(
            sizeof(chunk_map)
        )
        res.size = chunk1.size
        res.reads = <read_map **>libc.stdlib.malloc(sizeof(read_map *))
        for i in range(chunk1.size):
            res.reads[i] = self._align(chunk1.reads[i], chunk2.reads[i])
        return res
