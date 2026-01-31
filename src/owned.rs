use crate::{
    QueueError, pack_entry,
    traits::{QueueConsumer, QueueFactory, QueueProducer},
    unpack_entry,
};
use crossbeam_utils::CachePadded;
use portable_atomic::AtomicU128;
use std::{
    fmt,
    marker::PhantomData,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};
/// Max attempts to try and acquire a place in the queue (read or write)
const MAX_ATTEMPTS: usize = if cfg!(test) { 1000 } else { u16::MAX as usize };

/// Storage abstraction that can be either statically or dynamically sized.
///
/// This enum provides a unified interface for both compile-time sized arrays
/// (static allocation) and runtime-sized vectors (dynamic allocation). The
/// choice is determined by the const generic parameter `N`:
/// - `N = 0`: Dynamic allocation using `Box<[T]>`
/// - `N > 0`: Static allocation using `[T; N]`
///
/// This design enables zero-cost abstractions where the compiler can optimize
/// away the enum dispatch in most cases.
enum Storage<const N: usize> {
    /// Statically allocated array of cache-padded atomics.
    ///
    /// Used when `N > 0`. Memory is allocated on the stack or embedded directly
    /// in the containing structure, providing better cache locality and no heap
    /// allocation overhead.
    Static([CachePadded<AtomicU128>; N]),

    /// Dynamically allocated boxed slice of cache-padded atomics.
    ///
    /// Used when `N = 0`. Memory is heap-allocated at runtime, allowing for
    /// flexible queue capacities determined at construction time.
    Dynamic(Box<[CachePadded<AtomicU128>]>),
}

impl<const N: usize> Storage<N> {
    /// Load an atomic value from the storage at the given index.
    ///
    /// # Arguments
    ///
    /// * `idx` - The index of the element to load
    /// * `order` - The memory ordering for the atomic load operation
    ///
    /// # Returns
    ///
    /// The 128-bit value at the specified index
    #[inline]
    fn load(&self, idx: usize, order: Ordering) -> u128 {
        match self {
            Self::Static(a) => a[idx].load(order),
            Self::Dynamic(v) => v[idx].load(order),
        }
    }

    /// Atomically compare and exchange a value in the storage.
    ///
    /// Compares the current value at `idx` with `old`, and if they match,
    /// replaces it with `new`. This is the fundamental operation for lock-free
    /// algorithms.
    ///
    /// # Arguments
    ///
    /// * `idx` - The index of the element to update
    /// * `old` - The expected current value
    /// * `new` - The new value to write if the comparison succeeds
    /// * `success` - Memory ordering for successful exchange
    /// * `failure` - Memory ordering for failed exchange
    ///
    /// # Returns
    ///
    /// * `Ok(old)` - if the exchange succeeded,
    /// * `Err(actual)` - if it failed with the actual value found
    #[inline]
    fn compare_exchange(
        &self,
        idx: usize,
        old: u128,
        new: u128,
        success: Ordering,
        failure: Ordering,
    ) -> Result<u128, u128> {
        match self {
            Self::Static(a) => a[idx].compare_exchange(old, new, success, failure),
            Self::Dynamic(v) => v[idx].compare_exchange(old, new, success, failure),
        }
    }
}

/// Lock-free MPMC queue with configurable behavior.
///
/// This is the foundational queue implementation that stores `Copy` types using
/// packed 128-bit atomic operations. It provides:
///
/// - **Lock-free operations**: All push/pop operations use atomic CAS without
///   locks
/// - **MPMC support**: Multiple producers and consumers can operate
///   concurrently
/// - **Sequence tracking**: Built-in sequence numbers for ordering verification
/// - **Flexible sizing**: Supports both compile-time (N > 0) and runtime (N =
///   0) capacity
///
/// # Type Parameters
///
/// * `T` - The data type to store (must be `Copy + Send + Sync + Default`)
/// * `I` - The index/sequence number type (default: `u32`, must convert to
///   `u128`)
/// * `N` - Compile-time capacity (0 = dynamic allocation, >0 = static
///   allocation)
///
/// # Memory Layout
///
/// Each queue slot stores a packed 128-bit value containing:
/// - Upper bits: Data value of type `T`
/// - Lower bits: Sequence number of type `I`
///
/// This allows atomic updates of both data and sequence in a single operation.
///
/// # Constraints
///
/// - `sizeof(T) + sizeof(I) <= 16` bytes (enforced at construction)
/// - Capacity must be a power of 2 and >= 2
/// - `N` must match runtime capacity if N > 0
pub struct MpmcQueue<T, I = u32, const N: usize = 0>
where
    T: Copy + Send + Sync + Default,
    I: Copy + Into<u128>,
{
    storage: Storage<N>,
    capacity: usize,
    mask: usize,
    write_index: AtomicUsize,
    pub(crate) read_index: AtomicUsize,
    data_size: usize,
    seq_shift: u32,
    _phantom: PhantomData<(T, I)>,
}

impl<T, I, const N: usize> fmt::Debug for MpmcQueue<T, I, N>
where
    T: Copy + Send + Sync + Default + fmt::Debug,
    I: Copy + Into<u128> + fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MpmcQueue")
            .field("capacity", &self.capacity)
            .field("len", &self.len())
            .field("is_empty", &self.is_empty())
            .finish_non_exhaustive()
    }
}

/// Builder for creating MPMC queues with different configurations.
///
/// Provides a fluent API for constructing queues with validated parameters.
/// Supports both dynamic and static (compile-time) capacity configurations.
///
/// # Type Parameters
///
/// * `T` - The data type to store (must be `Copy + Send + Sync + Default`)
/// * `I` - The index/sequence number type (default: `u32`)
///
/// # Examples
///
/// ```
/// use jacques::{
///     owned::queue,
///     traits::{QueueConsumer, QueueProducer},
/// };
///
/// # fn main() -> Result<(), jacques::QueueError> {
/// // Create a queue with dynamic capacity
/// let (producer, consumer) = queue::<u32>().capacity(64).channels()?;
///
/// producer.push(100)?;
/// assert_eq!(consumer.pop()?, 100);
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct QueueBuilder<T, I = u32>
where
    T: Copy + Send + Sync + Default,
    I: Copy + Into<u128>,
{
    capacity: Option<usize>,
    _phantom: PhantomData<(T, I)>,
}

impl<T, I> Default for QueueBuilder<T, I>
where
    T: Copy + Send + Sync + Default,
    I: Copy + Into<u128>,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T, I> QueueBuilder<T, I>
where
    T: Copy + Send + Sync + Default,
    I: Copy + Into<u128>,
{
    /// Create a new queue builder
    pub const fn new() -> Self {
        Self {
            capacity: None,
            _phantom: PhantomData,
        }
    }

    /// Set the queue capacity (must be a power of 2)
    #[must_use]
    pub const fn capacity(mut self, cap: usize) -> Self {
        self.capacity = Some(cap);
        self
    }

    /// Build a dynamic queue
    pub fn build(self) -> Result<Arc<MpmcQueue<T, I>>, QueueError> {
        let capacity = self.capacity.ok_or(QueueError::InvalidCapacity)?;
        Ok(Arc::new(MpmcQueue::new(capacity)?))
    }

    /// Build a static queue with compile-time capacity
    pub fn build_static<const N: usize>(self) -> Result<Arc<MpmcQueue<T, I, N>>, QueueError> {
        let capacity = self.capacity.unwrap_or(N);
        Ok(Arc::new(MpmcQueue::new(capacity)?))
    }

    /// Create producer/consumer pair
    pub fn channels(self) -> Result<(Producer<T, I>, Consumer<T, I>), QueueError> {
        let queue = self.build()?;
        Ok((queue.producer(), queue.consumer()))
    }

    /// Create producer/consumer pair with static capacity
    pub fn channels_static<const N: usize>(
        self,
    ) -> Result<(Producer<T, I, N>, Consumer<T, I, N>), QueueError> {
        let queue = self.build_static::<N>()?;
        Ok((queue.producer(), queue.consumer()))
    }
}

/// Convenience function for creating queues with default index type (`u32`).
///
/// This is the primary entry point for creating MPMC queues. Returns a builder
/// that allows configuring capacity and other parameters.
///
/// # Examples
///
/// ```
/// use jacques::{owned::queue, traits::QueueProducer};
///
/// # fn main() -> Result<(), jacques::QueueError> {
/// let (producer, consumer) = queue::<u64>().capacity(128).channels()?;
///
/// producer.push(42)?;
/// # Ok(())
/// # }
/// ```
pub const fn queue<T>() -> QueueBuilder<T, u32>
where
    T: Copy + Send + Sync + Default,
{
    QueueBuilder::new()
}

/// Convenience function for creating queues with custom index type.
///
/// Use this when you need larger sequence numbers (e.g., `u64`) or want to
/// optimize memory usage with smaller types (e.g., `u16`).
///
/// # Type Parameters
///
/// * `T` - The data type to store
/// * `I` - The index/sequence number type
///
/// # Examples
///
/// ```
/// use jacques::{owned::queue_with_index, traits::QueueProducer};
///
/// # fn main() -> Result<(), jacques::QueueError> {
/// // Use u16 for smaller overhead when storing small types
/// let (producer, consumer) = queue_with_index::<u8, u16>().capacity(256).channels()?;
///
/// producer.push(42u8)?;
/// # Ok(())
/// # }
/// ```
pub const fn queue_with_index<T, I>() -> QueueBuilder<T, I>
where
    T: Copy + Send + Sync + Default,
    I: Copy + Into<u128>,
{
    QueueBuilder::new()
}

impl<T, I, const N: usize> MpmcQueue<T, I, N>
where
    T: Copy + Send + Sync + Default,
    I: Copy + Into<u128>,
{
    /// Create a new queue with the specified capacity
    pub(crate) fn new(mut cap: usize) -> Result<Self, QueueError> {
        // Validate capacity
        if N > 0 && cap != N {
            return Err(QueueError::CapacityMismatch);
        }

        cap = cap.max(2).next_power_of_two();

        let data_size = size_of::<T>();
        let index_size = size_of::<I>();

        // Ensure packed entry fits in u128
        if data_size + index_size > 16 {
            return Err(QueueError::TypeSizeExceeded { size: data_size });
        }

        let seq_shift = u32::try_from(index_size * 8).map_err(|_| QueueError::CapacityMismatch)?;
        let seq_mask = (1u128 << seq_shift) - 1u128;

        let storage = if N > 0 {
            // Static allocation
            Storage::Static(std::array::from_fn(|i| {
                let seq = ((i as u128) << 1) & seq_mask;
                let packed = pack_entry::<T, I>(seq, T::default(), seq_shift);
                CachePadded::new(AtomicU128::new(packed))
            }))
        } else {
            // Dynamic allocation
            Storage::Dynamic(
                (0..cap)
                    .map(|i| {
                        let seq = ((i as u128) << 1) & seq_mask;
                        let packed = pack_entry::<T, I>(seq, T::default(), seq_shift);
                        CachePadded::new(AtomicU128::new(packed))
                    })
                    .collect(),
            )
        };

        Ok(Self {
            storage,
            capacity: cap,
            mask: cap - 1,
            write_index: AtomicUsize::new(0),
            read_index: AtomicUsize::new(0),
            data_size,
            seq_shift,
            _phantom: PhantomData,
        })
    }

    /// Get the capacity of the queue
    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    /// Get the current number of elements in the queue
    pub fn len(&self) -> usize {
        let read = self.read_index.load(Ordering::Relaxed);
        let write = self.write_index.load(Ordering::Relaxed);
        write.wrapping_sub(read)
    }

    /// Check if the queue is empty
    pub fn is_empty(&self) -> bool {
        let rd = self.read_index.load(Ordering::Acquire);
        let idx = rd & self.mask;
        let packed = self.storage.load(idx, Ordering::Acquire);
        let (seq, _) = unpack_entry::<T, I>(packed, self.seq_shift, self.data_size);
        seq == ((rd as u128) << 1)
    }

    /// Check if the queue is full
    pub fn is_full(&self) -> bool {
        self.len() >= self.capacity
    }

    /// Atomically exchange a value at a specific index if it matches the
    /// expected value.
    ///
    /// This operation performs a compare-and-swap on a specific queue slot,
    /// useful for advanced use cases like in-place updates or custom
    /// synchronization patterns.
    ///
    /// # Arguments
    ///
    /// * `index` - The logical index to exchange (will be masked to actual
    ///   slot)
    /// * `old_value` - The expected current value
    /// * `new_value` - The new value to write if exchange succeeds
    ///
    /// # Returns
    ///
    /// `true` if the exchange succeeded, `false` if the current value didn't
    /// match
    ///
    /// # Note
    ///
    /// This is a low-level operation that bypasses normal queue semantics. The
    /// sequence number at the slot must be odd (indicating a filled slot)
    /// with the LSB set.
    pub fn exchange(&self, index: usize, old_value: T, new_value: T) -> bool {
        let idx = index & self.mask;
        let seq = ((index as u128) << 1) | 1u128;

        let old_packed = pack_entry::<T, I>(seq, old_value, self.seq_shift);
        let new_packed = pack_entry::<T, I>(seq, new_value, self.seq_shift);

        self.storage
            .compare_exchange(
                idx,
                old_packed,
                new_packed,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_ok()
    }

    /// Try to push a value without blocking
    pub fn try_push(&self, value: T) -> Result<(), (T, QueueError)> {
        match self.push_impl(value, false).map(|_| ()) {
            Ok(()) => Ok(()),
            Err(QueueError::Full) => Err((value, QueueError::Full)),
            Err(e) => Err((value, e)),
        }
    }

    /// Push a value, spinning until successful
    pub fn push(&self, value: T) -> Result<(), QueueError> {
        self.push_impl(value, true).map(|_| ())
    }

    /// Push with actual index returned
    pub(crate) fn push_impl(&self, value: T, retry: bool) -> Result<usize, QueueError> {
        let mut attempts = 0;

        loop {
            if !retry && attempts > 0 {
                return Err(QueueError::Full);
            }

            attempts += 1;
            if attempts > MAX_ATTEMPTS {
                return Err(QueueError::Full);
            }

            let wr = self.write_index.load(Ordering::Acquire);
            let idx = wr & self.mask;
            let packed = self.storage.load(idx, Ordering::Acquire);
            let (seq, _) = unpack_entry::<T, I>(packed, self.seq_shift, self.data_size);

            let expected_seq = (wr as u128) << 1;

            if seq == expected_seq {
                // Slot is available
                let new_seq = expected_seq | 1u128;
                let new_packed = pack_entry::<T, I>(new_seq, value, self.seq_shift);

                if self
                    .storage
                    .compare_exchange(idx, packed, new_packed, Ordering::AcqRel, Ordering::Acquire)
                    .is_ok()
                {
                    // Help advance write index
                    let _ = self.write_index.compare_exchange_weak(
                        wr,
                        wr + 1,
                        Ordering::AcqRel,
                        Ordering::Acquire,
                    );
                    return Ok(wr); // Return actual index used
                }
            } else if seq == (expected_seq | 1u128) {
                // Help advance write index
                let _ = self.write_index.compare_exchange_weak(
                    wr,
                    wr + 1,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                );
            } else if seq.wrapping_add((self.capacity as u128) << 1) == (expected_seq | 1u128) {
                // Queue is full
                if !retry {
                    return Err(QueueError::Full);
                }
                std::hint::spin_loop();
            }
        }
    }

    /// Try to pop a value without blocking
    pub fn try_pop(&self) -> Result<T, QueueError> {
        self.pop_impl(false).map(|(data, _)| data)
    }

    /// Pop a value, spinning until successful
    pub fn pop(&self) -> Result<T, QueueError> {
        self.pop_impl(true).map(|(data, _)| data)
    }

    /// Pop with actual index returned
    pub(crate) fn pop_impl(&self, retry: bool) -> Result<(T, usize), QueueError> {
        let mut attempts = 0;

        loop {
            if !retry && attempts > 0 {
                return Err(QueueError::Empty);
            }

            attempts += 1;
            if attempts > MAX_ATTEMPTS {
                return Err(QueueError::Empty);
            }

            let rd = self.read_index.load(Ordering::Acquire);
            let idx = rd & self.mask;
            let packed = self.storage.load(idx, Ordering::Acquire);
            let (seq, data) = unpack_entry::<T, I>(packed, self.seq_shift, self.data_size);

            let expected_full = ((rd as u128) << 1) | 1u128;

            if seq == expected_full {
                // Data is available
                let new_seq =
                    (((rd + self.capacity) as u128) << 1) & ((1u128 << self.seq_shift) - 1u128);
                let new_packed = pack_entry::<T, I>(new_seq, T::default(), self.seq_shift);

                if self
                    .storage
                    .compare_exchange(idx, packed, new_packed, Ordering::AcqRel, Ordering::Acquire)
                    .is_ok()
                {
                    // Help advance read index
                    let _ = self.read_index.compare_exchange_weak(
                        rd,
                        rd + 1,
                        Ordering::AcqRel,
                        Ordering::Acquire,
                    );
                    return Ok((data, rd)); // Return actual index used
                }
            } else if seq == ((rd as u128) << 1) {
                // Queue is empty
                if !retry {
                    return Err(QueueError::Empty);
                }
                std::hint::spin_loop();
            } else {
                // Help advance read index
                let _ = self.read_index.compare_exchange_weak(
                    rd,
                    rd + 1,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                );
            }
        }
    }

    /// Peek at the front element without removing it
    pub fn peek(&self) -> Result<T, QueueError> {
        let rd = self.read_index.load(Ordering::Acquire);
        let idx = rd & self.mask;
        let packed = self.storage.load(idx, Ordering::Acquire);
        let (seq, data) = unpack_entry::<T, I>(packed, self.seq_shift, self.data_size);

        if seq == (((rd as u128) << 1) | 1u128) {
            Ok(data)
        } else {
            Err(QueueError::Empty)
        }
    }
}

// Type aliases for common configurations

/// Convenient type alias for [`QueueProducerHandle`].
///
/// This simplifies the type signatures when using producer handles with default
/// configuration parameters.
pub type Producer<T, I = u32, const N: usize = 0> = QueueProducerHandle<T, I, N>;

/// Convenient type alias for [`QueueConsumerHandle`].
///
/// This simplifies the type signatures when using consumer handles with default
/// configuration parameters.
pub type Consumer<T, I = u32, const N: usize = 0> = QueueConsumerHandle<T, I, N>;

/// Producer handle for the MPMC queue.
///
/// A lightweight, cloneable handle that allows pushing items to the queue.
/// Multiple producer handles can be created for the same queue, enabling
/// multi-producer scenarios. Each clone shares the same underlying queue
/// via `Arc`.
///
/// # Type Parameters
///
/// * `T` - The data type to store (must be `Copy + Send + Sync + Default`)
/// * `I` - The index/sequence number type (default: `u32`)
/// * `N` - Compile-time capacity (0 = dynamic, >0 = static)
///
/// # Examples
///
/// ```
/// use jacques::{owned::queue, traits::QueueProducer};
/// use std::thread;
///
/// # fn main() -> Result<(), jacques::QueueError> {
/// let (producer, consumer) = queue::<u64>().capacity(128).channels()?;
///
/// // Clone producer for another thread
/// let producer2 = producer.clone();
/// let handle = thread::spawn(move || {
///     producer2.push(42).unwrap();
/// });
///
/// producer.push(100)?;
/// handle.join().unwrap();
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct QueueProducerHandle<T, I = u32, const N: usize = 0>
where
    T: Copy + Send + Sync + Default,
    I: Copy + Into<u128>,
{
    queue: Arc<MpmcQueue<T, I, N>>,
}

impl<T, I, const N: usize> Clone for QueueProducerHandle<T, I, N>
where
    T: Copy + Send + Sync + Default,
    I: Copy + Into<u128>,
{
    fn clone(&self) -> Self {
        Self {
            queue: self.queue.clone(),
        }
    }
}

impl<T, I, const N: usize> QueueProducer<T> for QueueProducerHandle<T, I, N>
where
    T: Copy + Send + Sync + Default,
    I: Copy + Into<u128>,
{
    fn try_push(&self, value: T) -> Result<(), (T, QueueError)> {
        self.queue.try_push(value)
    }
    fn push(&self, value: T) -> Result<(), QueueError> {
        self.queue.push(value)
    }

    fn push_with_seq(&self, value: T) -> Result<usize, QueueError> {
        self.queue.push_impl(value, true)
    }
}

/// Consumer handle for the MPMC queue.
///
/// A lightweight, cloneable handle that allows popping items from the queue.
/// Multiple consumer handles can be created for the same queue, enabling
/// multi-consumer scenarios. Each clone shares the same underlying queue
/// via `Arc`.
///
/// Provides rich consumption APIs including conditional popping (`pop_if`),
/// bulk consumption (`consume`), and peeking without removal.
///
/// # Type Parameters
///
/// * `T` - The data type to store (must be `Copy + Send + Sync + Default`)
/// * `I` - The index/sequence number type (default: `u32`)
/// * `N` - Compile-time capacity (0 = dynamic, >0 = static)
///
/// # Examples
///
/// ```
/// use jacques::{
///     owned::queue,
///     traits::{QueueConsumer, QueueProducer},
/// };
///
/// # fn main() -> Result<(), jacques::QueueError> {
/// let (producer, consumer) = queue::<i32>().capacity(64).channels()?;
///
/// producer.push(1)?;
/// producer.push(2)?;
/// producer.push(3)?;
///
/// // Conditional pop - only pop even numbers
/// if let Ok(value) = consumer.pop_if(|&v, _| v % 2 == 0) {
///     assert_eq!(value, 2);
/// }
///
/// // Bulk consume
/// let mut sum = 0;
/// consumer.consume(|val, _seq| {
///     sum += val;
///     false // continue until empty
/// });
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct QueueConsumerHandle<T, I = u32, const N: usize = 0>
where
    T: Copy + Send + Sync + Default,
    I: Copy + Into<u128>,
{
    queue: Arc<MpmcQueue<T, I, N>>,
}

impl<T, I, const N: usize> Clone for QueueConsumerHandle<T, I, N>
where
    T: Copy + Send + Sync + Default,
    I: Copy + Into<u128>,
{
    fn clone(&self) -> Self {
        Self {
            queue: self.queue.clone(),
        }
    }
}

impl<T, I, const N: usize> QueueConsumer<T> for QueueConsumerHandle<T, I, N>
where
    T: Copy + Send + Sync + Default,
    I: Copy + Into<u128>,
{
    fn try_pop(&self) -> Result<T, QueueError> {
        self.queue.try_pop()
    }
    fn pop(&self) -> Result<T, QueueError> {
        self.queue.pop()
    }

    fn pop_with_seq(&self) -> Result<(T, usize), QueueError> {
        self.queue.pop_impl(true)
    }

    fn peek(&self) -> Result<T, QueueError> {
        self.queue.peek()
    }

    fn peek_with_seq(&self) -> Result<(T, usize), QueueError> {
        let rd = self.queue.read_index.load(Ordering::Acquire);
        self.queue.peek().map(|value| (value, rd))
    }

    fn pop_if<F>(&self, mut predicate: F) -> Result<T, QueueError>
    where
        F: FnMut(&T, usize) -> bool,
    {
        loop {
            let rd = self.queue.read_index.load(Ordering::SeqCst);
            let idx = rd & self.queue.mask;
            let packed = self.queue.storage.load(idx, Ordering::SeqCst);
            let (seq, data) =
                unpack_entry::<T, I>(packed, self.queue.seq_shift, self.queue.data_size);

            if seq == ((rd as u128) << 1) {
                return Err(QueueError::Empty);
            }

            if seq == (((rd as u128) << 1) | 1u128) {
                if !predicate(&data, rd) {
                    return Err(QueueError::Empty); // predicate failed
                }

                let new_seq = (((rd + self.queue.capacity) as u128) << 1)
                    & ((1u128 << self.queue.seq_shift) - 1u128);
                let new_packed = pack_entry::<T, I>(new_seq, T::default(), self.queue.seq_shift);

                if self
                    .queue
                    .storage
                    .compare_exchange(idx, packed, new_packed, Ordering::SeqCst, Ordering::SeqCst)
                    .is_ok()
                {
                    let _ = self.queue.read_index.compare_exchange(
                        rd,
                        rd + 1,
                        Ordering::SeqCst,
                        Ordering::SeqCst,
                    );
                    return Ok(data);
                }
            } else if (seq >> 1) == ((rd + self.queue.capacity) as u128) {
                let _ = self.queue.read_index.compare_exchange(
                    rd,
                    rd + 1,
                    Ordering::SeqCst,
                    Ordering::SeqCst,
                );
            }
        }
    }

    fn consume<F>(&self, mut consumer: F) -> usize
    where
        F: FnMut(T, usize) -> bool,
    {
        let mut count = 0;
        while let Ok((value, seq)) = self.pop_with_seq() {
            count += 1;
            if consumer(value, seq) {
                break;
            }
        }
        count
    }

    fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    fn size(&self) -> usize {
        self.queue.len()
    }
}

impl<T, I, const N: usize> QueueFactory<T> for Arc<MpmcQueue<T, I, N>>
where
    T: Copy + Send + Sync + Default,
    I: Copy + Into<u128>,
{
    type Producer = QueueProducerHandle<T, I, N>;
    type Consumer = QueueConsumerHandle<T, I, N>;

    fn producer(&self) -> Self::Producer {
        QueueProducerHandle {
            queue: self.clone(),
        }
    }

    fn consumer(&self) -> Self::Consumer {
        QueueConsumerHandle {
            queue: self.clone(),
        }
    }
}

// Safety: The queue only contains atomic operations and Copy types
unsafe impl<T, I, const N: usize> Send for MpmcQueue<T, I, N>
where
    T: Copy + Send + Sync + Default,
    I: Copy + Into<u128>,
{
}

unsafe impl<T, I, const N: usize> Sync for MpmcQueue<T, I, N>
where
    T: Copy + Send + Sync + Default,
    I: Copy + Into<u128>,
{
}
#[cfg(test)]
mod tests {
    use super::*;

    // runtime capacity
    #[test]
    fn runtime_basic() {
        let q = queue::<u32>().capacity(8).build().unwrap();

        assert_eq!(q.capacity(), 8);
        assert_eq!(q.len(), 0);

        let (producer, consumer) = q.channel();
        producer.push(10).unwrap();
        assert_eq!(consumer.pop().unwrap(), 10);
    }

    // compile-time capacity
    #[test]
    fn static_basic() {
        let q = queue::<u32>().capacity(4).build().unwrap();
        assert_eq!(q.capacity(), 4);
        assert_eq!(q.len(), 0);

        let (producer, consumer) = q.channel();
        producer.push(7).unwrap();
        assert_eq!(consumer.pop().unwrap(), 7);
    }

    #[test]
    fn push_pop_wrap() {
        let (producer, consumer) = queue::<u32>().capacity(8).channels().unwrap();

        for i in 0..8 {
            producer.push(i).unwrap();
        }
        assert!(matches!(producer.push(99), Err(QueueError::Full)));
        for i in 0..8 {
            let v = consumer.pop().unwrap();
            assert_eq!(v, i);
        }
        assert!(matches!(consumer.pop(), Err(QueueError::Empty)));
    }

    #[test]
    fn test_with_seq_operations() {
        let (producer, consumer) = queue::<u32>().capacity(8).channels().unwrap();
        // Test push/pop with sequence numbers
        let seq1 = producer.push_with_seq(100).unwrap();
        let seq2 = producer.push_with_seq(200).unwrap();

        assert_eq!(seq1, 0);
        assert_eq!(seq2, 1);

        let (val1, pop_seq1) = consumer.pop_with_seq().unwrap();
        let (val2, pop_seq2) = consumer.pop_with_seq().unwrap();

        assert_eq!(val1, 100);
        assert_eq!(val2, 200);
        assert_eq!(pop_seq1, 0);
        assert_eq!(pop_seq2, 1);
    }

    #[test]
    fn test_peek_operations() {
        let (producer, consumer) = queue::<u32>().capacity(8).channels().unwrap();

        assert!(consumer.peek().is_err());

        producer.push(42).unwrap();
        assert_eq!(consumer.peek().unwrap(), 42);

        let (val, seq) = consumer.peek_with_seq().unwrap();
        assert_eq!(val, 42);
        assert_eq!(seq, 0);

        // Peek shouldn't consume
        assert_eq!(consumer.pop().unwrap(), 42);
        assert!(consumer.peek().is_err());
    }

    #[test]
    fn test_pop_if() {
        let (producer, consumer) = queue::<u32>().capacity(8).channels().unwrap();

        producer.push(10).unwrap();
        producer.push(20).unwrap();
        producer.push(30).unwrap();

        // Pop if value > 15 - should fail since head is 10
        assert!(consumer.pop_if(|&v, _seq| v > 15).is_err());

        // Pop if value > 5 - should succeed since head is 10
        let val = consumer.pop_if(|&v, _seq| v > 5).unwrap();
        assert_eq!(val, 10);

        // Now head is 20, pop if > 15 should work
        let val = consumer.pop_if(|&v, _seq| v > 15).unwrap();
        assert_eq!(val, 20);
    }

    #[test]
    fn test_consume() {
        let (producer, consumer) = queue::<u32>().capacity(8).channels().unwrap();

        for i in 0..5 {
            producer.push(i).unwrap();
        }

        let mut consumed = Vec::new();
        let count = consumer.consume(|val, _seq| {
            consumed.push(val);
            val == 2 // stop when we see value 2
        });

        assert_eq!(count, 3); // consumed 0, 1, 2
        assert_eq!(consumed, vec![0, 1, 2]);

        // Queue should still have 3, 4
        assert_eq!(consumer.pop().unwrap(), 3);
        assert_eq!(consumer.pop().unwrap(), 4);
        assert!(consumer.is_empty());
    }

    #[test]
    fn test_exchange() {
        let (producer, consumer) = queue::<u32>().capacity(4).channels().unwrap();

        producer.push(100).unwrap();
        producer.push(200).unwrap();

        // Exchange value at index 0 from 100 to 150
        assert!(producer.queue.exchange(0, 100, 150));

        // Try to exchange wrong old value - should fail
        assert!(!producer.queue.exchange(0, 100, 160));

        // Verify the exchange worked
        assert_eq!(consumer.pop().unwrap(), 150);
        assert_eq!(consumer.pop().unwrap(), 200);
    }

    use crate::traits::{QueueConsumer, QueueFactory, QueueProducer};
    use std::{
        collections::HashSet,
        sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
        },
        time::Instant,
    };
    use tokio::{
        task,
        time::{Duration, sleep},
    };

    /// Multi-producer / multi-consumer stress test for runtime (dynamic)
    /// capacity
    #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
    async fn mpmc_stress_dynamic() {
        // parameters - tweak these for longer runs
        let producers = 4usize;
        let consumers = 4usize;
        let items_per_producer = 100_000usize;
        let capacity = 1024usize; // power-of-two

        let total = producers * items_per_producer;

        // create queue: runtime variant (N == 0)
        let (producer, consumer) = queue::<u64>().capacity(capacity).channels().unwrap();

        // shared set to detect duplicates/loss; use tokio mutex (async friendly)
        let seen = Arc::new(tokio::sync::Mutex::new(HashSet::<u64>::with_capacity(
            total,
        )));
        let consumed = Arc::new(AtomicUsize::new(0));

        // spawn consumers
        let mut consumer_handles = Vec::with_capacity(consumers);
        for _ in 0..consumers {
            let seen_cl = seen.clone();
            let consumed_cl = consumed.clone();
            let total_cl = total;
            let consumer = consumer.clone();
            let h = task::spawn(async move {
                loop {
                    // stop when we've consumed everything
                    if consumed_cl.load(Ordering::SeqCst) >= total_cl {
                        break;
                    }
                    match consumer.pop() {
                        Ok(val) => {
                            // record and detect duplicates
                            let inserted = seen_cl.lock().await.insert(val);
                            assert!(inserted, "duplicate value observed: {val}");
                            consumed_cl.fetch_add(1, Ordering::SeqCst);
                        },
                        Err(QueueError::Empty) => {
                            // avoid busy spinning; yield to other tasks
                            task::yield_now().await;
                        },
                        Err(e) => {
                            panic!("unexpected queue error in consumer: {e:?}");
                        },
                    }
                }
            });
            consumer_handles.push(h);
        }

        // spawn producers
        let mut producer_handles = Vec::with_capacity(producers);
        let start = Instant::now();
        for pid in 0..producers {
            let producer = producer.clone();
            let h = task::spawn(async move {
                for i in 0..items_per_producer {
                    let val = ((pid as u64) << 32) | (i as u64);
                    // try until pushed
                    loop {
                        match producer.push(val) {
                            Ok(()) => break,
                            Err(QueueError::Full) => {
                                // back off a little
                                task::yield_now().await;
                            },
                            Err(e) => {
                                panic!("unexpected queue error in producer: {e:?}");
                            },
                        }
                    }
                }
            });
            producer_handles.push(h);
        }

        // wait producers to finish
        for h in producer_handles {
            h.await.expect("producer join");
        }

        // wait until all items are consumed
        while consumed.load(Ordering::SeqCst) < total {
            // give consumers a chance to finish
            sleep(Duration::from_millis(1)).await;
        }

        // join consumers
        for h in consumer_handles {
            h.await.expect("consumer join");
        }

        let elapsed = start.elapsed();
        let throughput = (total as f64) / elapsed.as_secs_f64();

        let seen_len = { seen.lock().await.len() };
        assert_eq!(seen_len, total, "expected all items consumed once");

        println!(
            "DYNAMIC test: producers={producers} consumers={consumers} items/producer={items_per_producer} capacity={capacity} => total={total} elapsed={elapsed:?} throughput={throughput:.0} ops/sec"
        );
    }

    /// Multi-producer / multi-consumer stress test for compile-time (static)
    /// capacity
    const CAP: usize = 1024usize;
    #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
    async fn mpmc_stress_static() {
        // parameters - tweak these for longer runs
        let producers = 4usize;
        let consumers = 4usize;
        let items_per_producer = 100_000usize;

        let total = producers * items_per_producer;

        // create queue: static variant (N == CAP)
        let q = queue::<u64>().capacity(CAP).build().unwrap();

        // shared set to detect duplicates/loss
        let seen = Arc::new(tokio::sync::Mutex::new(HashSet::<u64>::with_capacity(
            total,
        )));
        let consumed = Arc::new(AtomicUsize::new(0));

        // spawn consumers
        let mut consumer_handles = Vec::with_capacity(consumers);
        for _ in 0..consumers {
            let q_cl = q.clone();
            let seen_cl = seen.clone();
            let consumed_cl = consumed.clone();
            let total_cl = total;
            let h = task::spawn(async move {
                loop {
                    if consumed_cl.load(Ordering::SeqCst) >= total_cl {
                        break;
                    }
                    let (_, consumer) = q_cl.channel();
                    match consumer.pop() {
                        Ok(val) => {
                            let inserted = seen_cl.lock().await.insert(val);
                            assert!(inserted, "duplicate value observed: {val}");
                            consumed_cl.fetch_add(1, Ordering::SeqCst);
                        },
                        Err(QueueError::Empty) => {
                            task::yield_now().await;
                        },
                        Err(e) => {
                            panic!("unexpected queue error in consumer: {e:?}");
                        },
                    }
                }
            });
            consumer_handles.push(h);
        }

        // spawn producers
        let mut producer_handles = Vec::with_capacity(producers);
        let start = Instant::now();
        for pid in 0..producers {
            let q_cl = q.clone();
            let h = task::spawn(async move {
                for i in 0..items_per_producer {
                    let val = ((pid as u64) << 32) | (i as u64);
                    loop {
                        let (producer, _) = q_cl.channel();
                        match producer.push(val) {
                            Ok(()) => break,
                            Err(QueueError::Full) => {
                                task::yield_now().await;
                            },
                            Err(e) => {
                                panic!("unexpected queue error in producer: {e:?}");
                            },
                        }
                    }
                }
            });
            producer_handles.push(h);
        }

        // wait producers to finish
        for h in producer_handles {
            h.await.expect("producer join");
        }

        // wait until all items are consumed
        while consumed.load(Ordering::SeqCst) < total {
            sleep(Duration::from_millis(1)).await;
        }

        // join consumers
        for h in consumer_handles {
            h.await.expect("consumer join");
        }

        let elapsed = start.elapsed();
        let throughput = (total as f64) / elapsed.as_secs_f64();

        let seen_len = { seen.lock().await.len() };
        assert_eq!(seen_len, total, "expected all items consumed once");

        println!(
            "STATIC test: producers={producers} consumers={consumers} items/producer={items_per_producer} capacity={CAP} => total={total} elapsed={elapsed:?} throughput={throughput:.0} ops/sec"
        );
    }
}
