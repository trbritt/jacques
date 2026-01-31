use crate::{
    QueueError,
    owned::MpmcQueue,
    traits::{QueueConsumer, QueueFactory, QueueProducer},
};
use std::{
    fmt,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

/// A collection of MPMC queues that distributes load across multiple queues for
/// better performance.
///
/// `QueuePack` provides horizontal scaling by using multiple independent
/// queues, reducing contention and improving cache locality compared to a
/// single large queue. This design enables:
///
/// - **Reduced contention**: Multiple queues mean less thread competition
/// - **Better cache locality**: Producers stick to assigned queues
/// - **Automatic load balancing**: Consumers scan across all queues
/// - **Linear scaling**: Performance improves with more queues/cores
///
/// # Architecture
///
/// - **Producers**: Assigned to a specific queue via round-robin on creation
/// - **Consumers**: Start with a preferred queue, scan others when empty
/// - **Sequence numbers**: Encode queue index in high bits for global ordering
///
/// # Type Parameters
///
/// * `T` - The value type stored in the queues (must be `Copy + Send + Sync +
///   Default`)
/// * `I` - The index type for sequence numbers (default: `u32`)
/// * `G` - Number of queues in the pack (const generic, default: 4)
/// * `K` - Scan threshold parameter (default: 16, currently unused)
/// * `N` - Individual queue capacity (0 = dynamic, >0 = static, default: 0)
///
/// # Examples
///
/// ```
/// use jacques::{
///     pack::queue_pack,
///     traits::{QueueConsumer, QueueProducer},
/// };
///
/// # fn main() -> Result<(), jacques::QueueError> {
/// // Create a pack with 4 queues, each with capacity 256
/// let (producer, consumer) = queue_pack::<u64, 4, 16>().queue_capacity(256).channels()?;
///
/// // Producers are assigned to queues
/// producer.push(100)?;
///
/// // Consumers scan all queues
/// assert_eq!(consumer.pop()?, 100);
///
/// // Check pack statistics
/// let stats = consumer.scan_stats();
/// println!("Pack has {} queues", stats.len());
/// # Ok(())
/// # }
/// ```
pub struct QueuePack<T, I = u32, const G: usize = 4, const K: usize = 16, const N: usize = 0>
where
    T: Copy + Send + Sync + Default,
    I: Copy + Into<u128>,
{
    queues: Vec<Arc<MpmcQueue<T, I, N>>>,
    writer_counter: AtomicUsize,
    reader_counter: AtomicUsize,
}

impl<T, I, const G: usize, const K: usize, const N: usize> fmt::Debug for QueuePack<T, I, G, K, N>
where
    T: Copy + Send + Sync + Default + fmt::Debug,
    I: Copy + Into<u128> + fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("QueuePack")
            .field("queue_count", &G)
            .field("scan_threshold", &K)
            .field("queue_capacity", &self.queue_capacity())
            .field("total_len", &self.len())
            .field("total_capacity", &self.capacity())
            .finish()
    }
}

/// Builder for creating queue packs.
///
/// Provides a fluent API for constructing queue packs with validated
/// parameters. The builder allows configuring the capacity of individual queues
/// while the number of queues and scan threshold are specified via const
/// generics.
///
/// # Type Parameters
///
/// * `T` - The value type to store (must be `Copy + Send + Sync + Default`)
/// * `I` - The index type for sequence numbers (default: `u32`)
/// * `G` - Number of queues in the pack (const generic, default: 4)
/// * `K` - Scan threshold parameter (default: 16)
///
/// # Examples
///
/// ```
/// use jacques::{
///     pack::queue_pack,
///     traits::{QueueConsumer, QueueProducer},
/// };
///
/// # fn main() -> Result<(), jacques::QueueError> {
/// // Create a pack with 8 queues, each with capacity 512
/// let (producer, consumer) = queue_pack::<u32, 8, 16>().queue_capacity(512).channels()?;
///
/// producer.push(42)?;
/// assert_eq!(consumer.pop()?, 42);
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct QueuePackBuilder<T, I = u32, const G: usize = 4, const K: usize = 16>
where
    T: Copy + Send + Sync + Default,
    I: Copy + Into<u128>,
{
    queue_capacity: Option<usize>,
    _phantom: std::marker::PhantomData<(T, I)>,
}

impl<T, I, const G: usize, const K: usize> Default for QueuePackBuilder<T, I, G, K>
where
    T: Copy + Send + Sync + Default,
    I: Copy + Into<u128>,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T, I, const G: usize, const K: usize> QueuePackBuilder<T, I, G, K>
where
    T: Copy + Send + Sync + Default,
    I: Copy + Into<u128>,
{
    /// Create a new queue pack builder
    pub const fn new() -> Self {
        Self {
            queue_capacity: None,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Set the capacity of each individual queue
    #[must_use]
    pub const fn queue_capacity(mut self, capacity: usize) -> Self {
        self.queue_capacity = Some(capacity);
        self
    }

    /// Build a dynamic queue pack
    pub fn build(self) -> Result<Arc<QueuePack<T, I, G, K>>, QueueError> {
        let capacity = self.queue_capacity.ok_or(QueueError::InvalidCapacity)?;
        Ok(Arc::new(QueuePack::new(capacity)?))
    }

    /// Build a static queue pack with compile-time queue capacity
    pub fn build_static<const N: usize>(self) -> Result<Arc<QueuePack<T, I, G, K, N>>, QueueError> {
        let capacity = self.queue_capacity.unwrap_or(N);
        Ok(Arc::new(QueuePack::new(capacity)?))
    }

    /// Create producer/consumer pair
    pub fn channels(
        self,
    ) -> Result<(PackProducer<T, I, G, K>, PackConsumer<T, I, G, K>), QueueError> {
        let pack = self.build()?;
        Ok((pack.producer(), pack.consumer()))
    }

    /// Create producer/consumer pair with static capacity
    pub fn channels_static<const N: usize>(
        self,
    ) -> Result<(PackProducer<T, I, G, K, N>, PackConsumer<T, I, G, K, N>), QueueError> {
        let pack = self.build_static::<N>()?;
        Ok((pack.producer(), pack.consumer()))
    }
}

/// Convenience function for creating queue packs with default index type
/// (`u32`).
///
/// This is the primary entry point for creating queue packs. Returns a builder
/// that allows configuring individual queue capacity.
///
/// # Type Parameters
///
/// * `T` - The value type to store
/// * `G` - Number of queues in the pack (const generic)
/// * `K` - Scan threshold parameter (const generic)
///
/// # Examples
///
/// ```
/// use jacques::{
///     pack::queue_pack,
///     traits::{QueueConsumer, QueueProducer},
/// };
///
/// # fn main() -> Result<(), jacques::QueueError> {
/// // Create a pack with 4 queues, scan threshold 16
/// let (producer, consumer) = queue_pack::<i64, 4, 16>().queue_capacity(128).channels()?;
///
/// for i in 0..10 {
///     producer.push(i)?;
/// }
///
/// let sum: i64 = (0..10).map(|_| consumer.pop().unwrap()).sum();
/// assert_eq!(sum, 45);
/// # Ok(())
/// # }
/// ```
pub fn queue_pack<T, const G: usize, const K: usize>() -> QueuePackBuilder<T, u32, G, K>
where
    T: Copy + Send + Sync + Default,
{
    QueuePackBuilder::new()
}

/// Convenience function for creating queue packs with custom index type.
///
/// Use this when you need larger sequence numbers or want to optimize memory
/// usage.
///
/// # Type Parameters
///
/// * `T` - The value type to store
/// * `I` - The index type for sequence numbers
/// * `G` - Number of queues in the pack
/// * `K` - Scan threshold parameter
///
/// # Examples
///
/// ```
/// use jacques::{pack::queue_pack_with_index, traits::QueueProducer};
///
/// # fn main() -> Result<(), jacques::QueueError> {
/// // Use u64 for larger sequence space with 8 queues
/// let (producer, _consumer) = queue_pack_with_index::<u32, u64, 8, 16>()
///     .queue_capacity(256)
///     .channels()?;
///
/// producer.push(100)?;
/// # Ok(())
/// # }
/// ```
pub fn queue_pack_with_index<T, I, const G: usize, const K: usize>() -> QueuePackBuilder<T, I, G, K>
where
    T: Copy + Send + Sync + Default,
    I: Copy + Into<u128>,
{
    QueuePackBuilder::new()
}

impl<T, I, const G: usize, const K: usize, const N: usize> QueuePack<T, I, G, K, N>
where
    T: Copy + Send + Sync + Default,
    I: Copy + Into<u128>,
{
    /// Create a new queue pack with the specified individual queue capacity
    pub fn new(queue_capacity: usize) -> Result<Self, QueueError> {
        if G == 0 {
            return Err(QueueError::InvalidCapacity);
        }

        let mut queues = Vec::with_capacity(G);
        for _ in 0..G {
            queues.push(Arc::new(MpmcQueue::new(queue_capacity)?));
        }

        Ok(Self {
            queues,
            writer_counter: AtomicUsize::new(0),
            reader_counter: AtomicUsize::new(0),
        })
    }

    /// Get the number of queues in this pack
    pub const fn queue_count() -> usize {
        G
    }

    /// Get the scan threshold
    pub const fn scan_threshold() -> usize {
        K
    }

    /// Get the capacity of each individual queue
    pub fn queue_capacity(&self) -> usize {
        self.queues[0].capacity()
    }

    /// Get total capacity across all queues
    pub fn capacity(&self) -> usize {
        self.queues.len() * self.queue_capacity()
    }

    /// Get approximate total number of elements across all queues
    pub fn len(&self) -> usize {
        self.queues.iter().map(|q| q.len()).sum()
    }

    /// Check if all queues in the pack are empty
    pub fn is_empty(&self) -> bool {
        self.queues.iter().all(|q| q.is_empty())
    }

    /// Check if all queues in the pack are full
    pub fn is_full(&self) -> bool {
        self.queues.iter().all(|q| q.is_full())
    }

    /// Get statistics for each queue
    pub fn queue_stats(&self) -> Vec<QueueStats> {
        self.queues
            .iter()
            .enumerate()
            .map(|(index, queue)| QueueStats {
                index,
                len: queue.len(),
                capacity: queue.capacity(),
                is_empty: queue.is_empty(),
                is_full: queue.is_full(),
            })
            .collect()
    }

    /// Try to push to a specific queue by index
    pub fn try_push_to(&self, queue_index: usize, value: T) -> Result<(), (T, QueueError)> {
        if queue_index >= G {
            return Err((value, QueueError::InvalidCapacity));
        }
        self.queues[queue_index].try_push(value)
    }

    /// Try to pop from a specific queue by index
    pub fn try_pop_from(&self, queue_index: usize) -> Result<T, QueueError> {
        if queue_index >= G {
            return Err(QueueError::InvalidCapacity);
        }
        self.queues[queue_index].try_pop()
    }
}

/// Statistics for a single queue within the pack.
///
/// Provides a snapshot of a queue's current state, useful for monitoring
/// load distribution and debugging performance issues.
///
/// # Fields
///
/// * `index` - The index of this queue within the pack (0 to G-1)
/// * `len` - Current number of elements in this queue
/// * `capacity` - Maximum capacity of this queue
/// * `is_empty` - Whether this queue is currently empty
/// * `is_full` - Whether this queue is currently full
///
/// # Examples
///
/// ```
/// use jacques::{pack::queue_pack, traits::QueueProducer};
///
/// # fn main() -> Result<(), jacques::QueueError> {
/// let (producer, consumer) = queue_pack::<u32, 4, 16>().queue_capacity(128).channels()?;
///
/// // Add some items
/// for i in 0..50 {
///     producer.push(i)?;
/// }
///
/// // Get statistics for all queues
/// let pack = queue_pack::<u32, 4, 16>().queue_capacity(128).build()?;
/// let stats = pack.queue_stats();
///
/// for stat in stats {
///     println!(
///         "Queue {}: {}/{} items ({}% full)",
///         stat.index,
///         stat.len,
///         stat.capacity,
///         (stat.len * 100) / stat.capacity
///     );
/// }
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QueueStats {
    /// The index of this queue within the pack (0 to G-1)
    pub index: usize,
    /// Current number of elements in this queue
    pub len: usize,
    /// Maximum capacity of this queue
    pub capacity: usize,
    /// Whether this queue is currently empty
    pub is_empty: bool,
    /// Whether this queue is currently full
    pub is_full: bool,
}

// Type aliases for common configurations

/// Convenient type alias for [`PackProducerHandle`].
///
/// This simplifies the type signatures when using pack producer handles
/// with default configuration parameters (4 queues, scan every 16 ops, dynamic
/// capacity).
pub type PackProducer<T, I = u32, const G: usize = 4, const K: usize = 16, const N: usize = 0> =
    PackProducerHandle<T, I, G, K, N>;

/// Convenient type alias for [`PackConsumerHandle`].
///
/// This simplifies the type signatures when using pack consumer handles
/// with default configuration parameters (4 queues, scan every 16 ops, dynamic
/// capacity).
pub type PackConsumer<T, I = u32, const G: usize = 4, const K: usize = 16, const N: usize = 0> =
    PackConsumerHandle<T, I, G, K, N>;

/// Producer handle for the queue pack.
///
/// Each producer is assigned to a specific queue for optimal cache locality,
/// reducing contention. Producers are assigned via round-robin when created,
/// distributing load evenly across all queues in the pack.
///
/// # Assignment Strategy
///
/// - Producers get a dedicated queue index on creation
/// - All pushes go to the assigned queue
/// - Cloning creates a new producer with a different assignment
/// - This provides excellent cache locality and minimal contention
///
/// # Type Parameters
///
/// * `T` - The value type to store (must be `Copy + Send + Sync + Default`)
/// * `I` - The index type for sequence numbers (default: `u32`)
/// * `G` - Number of queues in the pack (default: 4)
/// * `K` - Scan threshold parameter (default: 16)
/// * `N` - Individual queue capacity (0 = dynamic, >0 = static, default: 0)
///
/// # Examples
///
/// ```
/// use jacques::{pack::queue_pack, traits::QueueProducer};
/// use std::thread;
///
/// # fn main() -> Result<(), jacques::QueueError> {
/// let (producer, consumer) = queue_pack::<u64, 4, 16>().queue_capacity(128).channels()?;
///
/// // Each clone gets assigned to a different queue
/// let producers: Vec<_> = (0..4).map(|_| producer.clone()).collect();
///
/// // Spawn threads, each with its own assigned queue
/// let handles: Vec<_> = producers
///     .into_iter()
///     .enumerate()
///     .map(|(id, p)| {
///         thread::spawn(move || {
///             for i in 0..100 {
///                 p.push((id as u64) * 1000 + i).unwrap();
///             }
///         })
///     })
///     .collect();
///
/// for h in handles {
///     h.join().unwrap();
/// }
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct PackProducerHandle<
    T,
    I = u32,
    const G: usize = 4,
    const K: usize = 16,
    const N: usize = 0,
> where
    T: Copy + Send + Sync + Default,
    I: Copy + Into<u128>,
{
    pack: Arc<QueuePack<T, I, G, K, N>>,
    queue_index: usize,
}

impl<T, I, const G: usize, const K: usize, const N: usize> Clone
    for PackProducerHandle<T, I, G, K, N>
where
    T: Copy + Send + Sync + Default,
    I: Copy + Into<u128>,
{
    fn clone(&self) -> Self {
        // Create a new producer with a different queue assignment
        self.pack.producer()
    }
}

impl<T, I, const G: usize, const K: usize, const N: usize> PackProducerHandle<T, I, G, K, N>
where
    T: Copy + Send + Sync + Default,
    I: Copy + Into<u128>,
{
    const fn new(pack: Arc<QueuePack<T, I, G, K, N>>, queue_index: usize) -> Self {
        Self { pack, queue_index }
    }

    /// Get the index of the queue this producer writes to
    pub const fn queue_index(&self) -> usize {
        self.queue_index
    }

    /// Try to push without blocking
    pub fn try_push(&self, value: T) -> Result<(), (T, QueueError)> {
        self.pack.queues[self.queue_index].try_push(value)
    }

    /// Get statistics for the assigned queue
    pub fn queue_stats(&self) -> QueueStats {
        let queue = &self.pack.queues[self.queue_index];
        QueueStats {
            index: self.queue_index,
            len: queue.len(),
            capacity: queue.capacity(),
            is_empty: queue.is_empty(),
            is_full: queue.is_full(),
        }
    }
}

impl<T, I, const G: usize, const K: usize, const N: usize> QueueProducer<T>
    for PackProducerHandle<T, I, G, K, N>
where
    T: Copy + Send + Sync + Default,
    I: Copy + Into<u128>,
{
    fn try_push(&self, value: T) -> Result<(), (T, QueueError)> {
        self.pack.queues[self.queue_index].try_push(value)
    }
    fn push(&self, value: T) -> Result<(), QueueError> {
        self.pack.queues[self.queue_index].push(value)
    }

    fn push_with_seq(&self, value: T) -> Result<usize, QueueError> {
        // Get the actual index from the underlying queue
        let local_index = self.pack.queues[self.queue_index].push_impl(value, true)?;

        // Encode: high 8 bits = queue index, low 24 bits = local index
        let global_seq = (self.queue_index << 24) | (local_index & 0x00FF_FFFF);
        Ok(global_seq)
    }
}

/// Consumer handle for the queue pack.
///
/// Consumers scan across queues to find work, providing automatic load
/// balancing. They maintain affinity to a preferred queue but will search all
/// other queues when the preferred queue is empty, ensuring work is always
/// found if available.
///
/// # Scanning Strategy
///
/// 1. Try the preferred queue first (best cache locality)
/// 2. If empty, scan all other queues in round-robin order
/// 3. Track operation count for statistics and sequence generation
///
/// This provides excellent load balancing while maintaining cache efficiency
/// when possible.
///
/// # Type Parameters
///
/// * `T` - The value type to store (must be `Copy + Send + Sync + Default`)
/// * `I` - The index type for sequence numbers (default: `u32`)
/// * `G` - Number of queues in the pack (default: 4)
/// * `K` - Scan threshold parameter (default: 16)
/// * `N` - Individual queue capacity (0 = dynamic, >0 = static, default: 0)
///
/// # Examples
///
/// ```
/// use jacques::{
///     pack::queue_pack,
///     traits::{QueueConsumer, QueueProducer},
/// };
///
/// # fn main() -> Result<(), jacques::QueueError> {
/// let (producer, consumer) = queue_pack::<u32, 4, 16>().queue_capacity(1024).channels()?;
///
/// // Add items to various queues
/// for i in 0..1000 {
///     producer.push(i)?;
/// }
///
/// // Consumer automatically scans all queues
/// let mut sum = 0;
/// for _ in 0..1000 {
///     sum += consumer.pop()?;
/// }
/// assert_eq!(sum, (0..1000).sum());
///
/// // Check statistics
/// let stats = consumer.scan_stats();
/// for stat in stats {
///     println!("Queue {}: {} items", stat.index, stat.len);
/// }
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct PackConsumerHandle<
    T,
    I = u32,
    const G: usize = 4,
    const K: usize = 16,
    const N: usize = 0,
> where
    T: Copy + Send + Sync + Default,
    I: Copy + Into<u128>,
{
    pack: Arc<QueuePack<T, I, G, K, N>>,
    preferred_queue_index: AtomicUsize,
    pop_count: AtomicUsize,
}

impl<T, I, const G: usize, const K: usize, const N: usize> Clone
    for PackConsumerHandle<T, I, G, K, N>
where
    T: Copy + Send + Sync + Default,
    I: Copy + Into<u128>,
{
    fn clone(&self) -> Self {
        // Create a new consumer with a different preferred queue
        self.pack.consumer()
    }
}

impl<T, I, const G: usize, const K: usize, const N: usize> PackConsumerHandle<T, I, G, K, N>
where
    T: Copy + Send + Sync + Default,
    I: Copy + Into<u128>,
{
    const fn new(pack: Arc<QueuePack<T, I, G, K, N>>, preferred_queue_index: usize) -> Self {
        Self {
            pack,
            preferred_queue_index: AtomicUsize::new(preferred_queue_index),
            pop_count: AtomicUsize::new(0),
        }
    }

    /// Get the preferred queue index for this consumer
    pub fn preferred_queue_index(&self) -> usize {
        self.preferred_queue_index.load(Ordering::Relaxed)
    }

    /// Get the current `pop_count` count
    pub fn pop_count(&self) -> usize {
        self.pop_count.load(Ordering::Relaxed)
    }

    /// Try to pop without blocking from preferred queue first, then scan others
    pub fn try_pop(&self) -> Result<T, QueueError> {
        let queue_idx = self.preferred_queue_index.load(Ordering::Relaxed);

        // Try preferred queue
        match self.pack.queues[queue_idx].try_pop() {
            Ok(value) => {
                let count = self.pop_count.fetch_add(1, Ordering::Relaxed) + 1;

                // After K successful pops, rotate to next queue
                if count >= K {
                    self.pop_count.store(0, Ordering::Relaxed);
                    let next_idx = (queue_idx + 1) % G;
                    self.preferred_queue_index
                        .store(next_idx, Ordering::Relaxed);
                }

                return Ok(value);
            },
            Err(QueueError::Empty) => {
                // Reset counter on empty
                self.pop_count.store(0, Ordering::Relaxed);
            },
            Err(e) => return Err(e),
        }

        // Scan other queues
        for i in 1..G {
            let scan_idx = (queue_idx + i) % G;
            match self.pack.queues[scan_idx].try_pop() {
                Ok(value) => {
                    // Found work in a different queue, switch to it
                    self.preferred_queue_index
                        .store(scan_idx, Ordering::Relaxed);
                    self.pop_count.store(1, Ordering::Relaxed);
                    return Ok(value);
                },
                Err(QueueError::Empty) => {},
                Err(e) => return Err(e),
            }
        }

        Err(QueueError::Empty)
    }

    /// Scan all queues and return statistics
    pub fn scan_stats(&self) -> Vec<QueueStats> {
        self.pack.queue_stats()
    }
}

impl<T, I, const G: usize, const K: usize, const N: usize> QueueConsumer<T>
    for PackConsumerHandle<T, I, G, K, N>
where
    T: Copy + Send + Sync + Default,
    I: Copy + Into<u128>,
{
    fn try_pop(&self) -> Result<T, QueueError> {
        let queue_idx = self.preferred_queue_index.load(Ordering::Relaxed);
        self.pack.queues[queue_idx].try_pop()
    }

    fn pop(&self) -> Result<T, QueueError> {
        let queue_idx = self.preferred_queue_index.load(Ordering::Relaxed);

        // Try preferred queue
        match self.pack.queues[queue_idx].pop() {
            Ok(value) => {
                let count = self.pop_count.fetch_add(1, Ordering::Relaxed) + 1;

                // After K successful pops, rotate to next queue
                if count >= K {
                    self.pop_count.store(0, Ordering::Relaxed);
                    let next_idx = (queue_idx + 1) % G;
                    self.preferred_queue_index
                        .store(next_idx, Ordering::Relaxed);
                }

                return Ok(value);
            },
            Err(QueueError::Empty) => {
                // Reset counter on empty
                self.pop_count.store(0, Ordering::Relaxed);
            },
            Err(e) => return Err(e),
        }

        // Scan other queues
        for i in 1..G {
            let scan_idx = (queue_idx + i) % G;
            match self.pack.queues[scan_idx].pop() {
                Ok(value) => {
                    // Found work in a different queue, switch to it
                    self.preferred_queue_index
                        .store(scan_idx, Ordering::Relaxed);
                    self.pop_count.store(1, Ordering::Relaxed);
                    return Ok(value);
                },
                Err(QueueError::Empty) => {},
                Err(e) => return Err(e),
            }
        }

        Err(QueueError::Empty)
    }

    fn pop_with_seq(&self) -> Result<(T, usize), QueueError> {
        let queue_idx = self.preferred_queue_index.load(Ordering::Relaxed);

        // Try preferred queue
        match self.pack.queues[queue_idx].pop_impl(true) {
            Ok((value, local_index)) => {
                let count = self.pop_count.fetch_add(1, Ordering::Relaxed) + 1;

                // After K successful pops, rotate to next queue
                if count >= K {
                    self.pop_count.store(0, Ordering::Relaxed);
                    let next_idx = (queue_idx + 1) % G;
                    self.preferred_queue_index
                        .store(next_idx, Ordering::Relaxed);
                }

                // Encode: high 8 bits = queue index, low 24 bits = local index
                let global_seq = (queue_idx << 24) | (local_index & 0x00FF_FFFF);
                return Ok((value, global_seq));
            },
            Err(QueueError::Empty) => {
                self.pop_count.store(0, Ordering::Relaxed);
            },
            Err(e) => return Err(e),
        }

        // Scan other queues
        for i in 1..G {
            let scan_idx = (queue_idx + i) % G;
            match self.pack.queues[scan_idx].pop_impl(true) {
                Ok((value, local_index)) => {
                    self.preferred_queue_index
                        .store(scan_idx, Ordering::Relaxed);
                    self.pop_count.store(1, Ordering::Relaxed);

                    let global_seq = (scan_idx << 24) | (local_index & 0x00FF_FFFF);
                    return Ok((value, global_seq));
                },
                Err(QueueError::Empty) => {},
                Err(e) => return Err(e),
            }
        }

        Err(QueueError::Empty)
    }

    fn peek(&self) -> Result<T, QueueError> {
        let queue_idx = self.preferred_queue_index.load(Ordering::Relaxed);

        // Try preferred queue first
        match self.pack.queues[queue_idx].peek() {
            Ok(value) => return Ok(value),
            Err(QueueError::Empty) => {},
            Err(e) => return Err(e),
        }

        // Scan other queues
        for i in 1..G {
            let scan_idx = (queue_idx + i) % G;
            match self.pack.queues[scan_idx].peek() {
                Ok(value) => return Ok(value),
                Err(QueueError::Empty) => {},
                Err(e) => return Err(e),
            }
        }

        Err(QueueError::Empty)
    }

    fn peek_with_seq(&self) -> Result<(T, usize), QueueError> {
        let queue_idx = self.preferred_queue_index.load(Ordering::Relaxed);

        // Try preferred queue first
        // Note: We can't get the actual sequence from peek since it doesn't modify
        // state, so we use the queue index in high bits and 0 in low bits as an
        // approximation
        match self.pack.queues[queue_idx].peek() {
            Ok(value) => {
                // Use queue index only, since we don't have access to actual slot index from
                // peek
                let seq = queue_idx << 24;
                return Ok((value, seq));
            },
            Err(QueueError::Empty) => {},
            Err(e) => return Err(e),
        }

        // Scan other queues
        for i in 1..G {
            let scan_idx = (queue_idx + i) % G;
            match self.pack.queues[scan_idx].peek() {
                Ok(value) => {
                    let seq = scan_idx << 24;
                    return Ok((value, seq));
                },
                Err(QueueError::Empty) => {},
                Err(e) => return Err(e),
            }
        }

        Err(QueueError::Empty)
    }

    fn pop_if<F>(&self, mut predicate: F) -> Result<T, QueueError>
    where
        F: FnMut(&T, usize) -> bool,
    {
        let queue_idx = self.preferred_queue_index.load(Ordering::Relaxed);

        // Try preferred queue first
        match self.pack.queues[queue_idx].peek() {
            Ok(value) => {
                // Use queue index only for peek sequence
                let seq = queue_idx << 24;
                if predicate(&value, seq) {
                    // Try to pop - if it succeeds, apply K-rotation logic
                    match self.pack.queues[queue_idx].pop() {
                        Ok(popped) => {
                            let count = self.pop_count.fetch_add(1, Ordering::Relaxed) + 1;

                            // After K successful pops, rotate to next queue
                            if count >= K {
                                self.pop_count.store(0, Ordering::Relaxed);
                                let next_idx = (queue_idx + 1) % G;
                                self.preferred_queue_index
                                    .store(next_idx, Ordering::Relaxed);
                            }

                            return Ok(popped);
                        },
                        Err(QueueError::Empty) => {
                            // Item was consumed by another thread between peek and pop
                            self.pop_count.store(0, Ordering::Relaxed);
                        },
                        Err(e) => return Err(e),
                    }
                }
                // Predicate failed, continue scanning other queues
            },
            Err(QueueError::Empty) => {
                self.pop_count.store(0, Ordering::Relaxed);
            },
            Err(e) => return Err(e),
        }

        // Scan other queues
        for i in 1..G {
            let scan_idx = (queue_idx + i) % G;
            match self.pack.queues[scan_idx].peek() {
                Ok(value) => {
                    let seq = scan_idx << 24;
                    if predicate(&value, seq) {
                        match self.pack.queues[scan_idx].pop() {
                            Ok(popped) => {
                                // Found work in different queue, switch to it
                                self.preferred_queue_index
                                    .store(scan_idx, Ordering::Relaxed);
                                self.pop_count.store(1, Ordering::Relaxed);
                                return Ok(popped);
                            },
                            Err(QueueError::Empty) => {},
                            Err(e) => return Err(e),
                        }
                    }
                },
                Err(QueueError::Empty) => {},
                Err(e) => return Err(e),
            }
        }

        Err(QueueError::Empty)
    }

    fn consume<F>(&self, mut consumer_fn: F) -> usize
    where
        F: FnMut(T, usize) -> bool,
    {
        let queue_idx = self.preferred_queue_index.load(Ordering::Relaxed);
        let mut total_consumed = 0;

        // Consume from all queues, starting with preferred
        for i in 0..G {
            let scan_idx = (queue_idx + i) % G;

            while let Ok((value, local_index)) = self.pack.queues[scan_idx].pop_impl(true) {
                // Encode queue index in high bits, local index in low bits
                let global_seq = (scan_idx << 24) | (local_index & 0x00FF_FFFF);
                total_consumed += 1;

                // Update preferred queue and pop count based on which queue we're consuming
                // from
                if scan_idx == queue_idx {
                    let count = self.pop_count.fetch_add(1, Ordering::Relaxed) + 1;
                    if count >= K {
                        self.pop_count.store(0, Ordering::Relaxed);
                        let next_idx = (queue_idx + 1) % G;
                        self.preferred_queue_index
                            .store(next_idx, Ordering::Relaxed);
                    }
                } else {
                    // Consuming from a different queue, switch to it
                    self.preferred_queue_index
                        .store(scan_idx, Ordering::Relaxed);
                    self.pop_count.store(1, Ordering::Relaxed);
                }

                if consumer_fn(value, global_seq) {
                    return total_consumed;
                }
            }
        }

        total_consumed
    }

    fn is_empty(&self) -> bool {
        self.pack.is_empty()
    }

    fn size(&self) -> usize {
        self.pack.len()
    }
}

impl<T, I, const G: usize, const K: usize, const N: usize> QueueFactory<T>
    for Arc<QueuePack<T, I, G, K, N>>
where
    T: Copy + Send + Sync + Default,
    I: Copy + Into<u128>,
{
    type Producer = PackProducerHandle<T, I, G, K, N>;
    type Consumer = PackConsumerHandle<T, I, G, K, N>;

    fn producer(&self) -> Self::Producer {
        let queue_index = self.writer_counter.fetch_add(1, Ordering::Relaxed) % G;
        PackProducerHandle::new(self.clone(), queue_index)
    }

    fn consumer(&self) -> Self::Consumer {
        let preferred_index = self.reader_counter.fetch_add(1, Ordering::Relaxed) % G;
        PackConsumerHandle::new(self.clone(), preferred_index)
    }
}

// Safety: The pack only contains Send+Sync queues and atomic counters
unsafe impl<T, I, const G: usize, const K: usize, const N: usize> Send for QueuePack<T, I, G, K, N>
where
    T: Copy + Send + Sync + Default,
    I: Copy + Into<u128>,
{
}

unsafe impl<T, I, const G: usize, const K: usize, const N: usize> Sync for QueuePack<T, I, G, K, N>
where
    T: Copy + Send + Sync + Default,
    I: Copy + Into<u128>,
{
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        collections::HashSet,
        sync::atomic::{AtomicUsize, Ordering},
        time::Instant,
    };

    #[test]
    fn test_builder_pattern() {
        let pack = queue_pack::<u32, 4, 16>()
            .queue_capacity(32)
            .build()
            .unwrap();

        assert_eq!(pack.queue_capacity(), 32);
        assert_eq!(pack.capacity(), 128); // 4 * 32
    }

    #[test]
    fn test_channels() {
        let (producer, consumer) = queue_pack::<u64, 2, 8>()
            .queue_capacity(16)
            .channels()
            .unwrap();

        producer.push(42).unwrap();
        producer.push(43).unwrap();

        assert_eq!(consumer.pop().unwrap(), 42);
        assert_eq!(consumer.pop().unwrap(), 43);
        assert!(consumer.try_pop().is_err());
    }

    #[test]
    fn test_producer_assignment() {
        let pack = queue_pack::<u32, 3, 4>().queue_capacity(8).build().unwrap();

        let p1 = pack.producer();
        let p2 = pack.producer();
        let p3 = pack.producer();
        let p4 = pack.producer(); // Should wrap around

        // Producers should be assigned to different queues in round-robin
        assert_eq!(p1.queue_index(), 0);
        assert_eq!(p2.queue_index(), 1);
        assert_eq!(p3.queue_index(), 2);
        assert_eq!(p4.queue_index(), 0); // Wrapped around
    }

    #[test]
    fn test_consumer_scanning() {
        let pack = queue_pack::<u64, 3, 2>().queue_capacity(8).build().unwrap();

        // Create producers for different queues
        let p1 = pack.producer(); // queue 0
        let p2 = pack.producer(); // queue 1
        let p3 = pack.producer(); // queue 2

        let consumer = pack.consumer();

        // Fill different queues
        p1.push(100).unwrap();
        p2.push(200).unwrap();
        p3.push(300).unwrap();

        // Consumer should find all values by scanning
        let mut values = Vec::new();
        for _ in 0..3 {
            values.push(consumer.pop().unwrap());
        }
        values.sort_unstable();
        assert_eq!(values, vec![100, 200, 300]);
    }

    #[test]
    fn test_queue_stats() {
        let pack = queue_pack::<u32, 2, 4>().queue_capacity(4).build().unwrap();

        let p1 = pack.producer(); // queue 0
        let p2 = pack.producer(); // queue 1

        // Add different amounts to each queue
        p1.push(1).unwrap();
        p1.push(2).unwrap();

        p2.push(10).unwrap();

        let stats = pack.queue_stats();
        assert_eq!(stats.len(), 2);

        assert_eq!(stats[0].index, 0);
        assert_eq!(stats[0].len, 2);
        assert_eq!(stats[0].capacity, 4);
        assert!(!stats[0].is_empty);
        assert!(!stats[0].is_full);

        assert_eq!(stats[1].index, 1);
        assert_eq!(stats[1].len, 1);
        assert_eq!(stats[1].capacity, 4);
        assert!(!stats[1].is_empty);
        assert!(!stats[1].is_full);
    }

    #[test]
    fn test_sequence_numbers() {
        let (producer, consumer) = queue_pack::<u32, 2, 4>()
            .queue_capacity(8)
            .channels()
            .unwrap();

        let seq1 = producer.push_with_seq(777).unwrap();
        let seq2 = producer.push_with_seq(888).unwrap();

        let (val1, _) = consumer.pop_with_seq().unwrap();
        let (val2, _) = consumer.pop_with_seq().unwrap();

        assert_eq!(val1, 777);
        assert_eq!(val2, 888);

        // Sequence numbers should encode queue index in high bits
        assert_eq!(seq1 >> 24, producer.queue_index());
        assert_eq!(seq2 >> 24, producer.queue_index());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
    async fn stress_test_pack() {
        const QUEUE_COUNT: usize = 4;
        const SCAN_THRESHOLD: usize = 16;
        const PRODUCERS: usize = 4;
        const CONSUMERS: usize = 4;
        const ITEMS_PER_PRODUCER: usize = 10_000;

        let (producer, consumer) = queue_pack::<u64, QUEUE_COUNT, SCAN_THRESHOLD>()
            .queue_capacity(256)
            .channels()
            .unwrap();

        let total_items = PRODUCERS * ITEMS_PER_PRODUCER;
        let consumed_count = Arc::new(AtomicUsize::new(0));
        let seen = Arc::new(tokio::sync::Mutex::new(HashSet::<u64>::new()));

        // Spawn consumers
        let mut consumer_handles = Vec::new();
        for _ in 0..CONSUMERS {
            let consumer = consumer.clone();
            let consumed_clone = consumed_count.clone();
            let seen_clone = seen.clone();

            let handle = tokio::task::spawn(async move {
                loop {
                    if consumed_clone.load(Ordering::SeqCst) >= total_items {
                        break;
                    }

                    match consumer.try_pop() {
                        Ok(value) => {
                            // Check for duplicates
                            assert!(
                                seen_clone.lock().await.insert(value),
                                "Duplicate value found: {value}"
                            );
                            consumed_clone.fetch_add(1, Ordering::SeqCst);
                        },
                        Err(QueueError::Empty) => {
                            tokio::task::yield_now().await;
                        },
                        Err(e) => panic!("Unexpected error: {e:?}"),
                    }
                }
            });
            consumer_handles.push(handle);
        }

        // Spawn producers
        let mut producer_handles = Vec::new();
        let start = Instant::now();

        for producer_id in 0..PRODUCERS {
            let producer = producer.clone();

            let handle = tokio::task::spawn(async move {
                for item_id in 0..ITEMS_PER_PRODUCER {
                    let value = ((producer_id as u64) << 32) | (item_id as u64);

                    loop {
                        match producer.try_push(value) {
                            Ok(()) => break,
                            Err((_, QueueError::Full)) => {
                                tokio::task::yield_now().await;
                            },
                            Err((_, e)) => panic!("Unexpected error: {e:?}"),
                        }
                    }
                }
            });
            producer_handles.push(handle);
        }

        // Wait for producers
        for handle in producer_handles {
            handle.await.unwrap();
        }

        // Wait for all items to be consumed
        while consumed_count.load(Ordering::SeqCst) < total_items {
            tokio::time::sleep(tokio::time::Duration::from_millis(1)).await;
        }

        // Wait for consumers
        for handle in consumer_handles {
            handle.await.unwrap();
        }

        let elapsed = start.elapsed();
        let throughput = (total_items as f64) / elapsed.as_secs_f64();

        println!(
            "Pack stress test: {QUEUE_COUNT} queues, {PRODUCERS} producers, {CONSUMERS} consumers, {ITEMS_PER_PRODUCER} items = {total_items} total in {elapsed:?} ({throughput:.0} ops/sec)"
        );

        assert_eq!(consumed_count.load(Ordering::SeqCst), total_items);
        let final_seen_count = seen.lock().await.len();
        assert_eq!(final_seen_count, total_items);
    }

    #[test]
    fn test_pop_if() {
        let (producer, consumer) = queue_pack::<u32, 2, 4>()
            .queue_capacity(8)
            .channels()
            .unwrap();

        producer.push(2).unwrap();
        producer.push(1).unwrap();
        producer.push(3).unwrap();

        // Pop only even numbers
        let result = consumer.pop_if(|&value, _seq| value % 2 == 0);
        assert_eq!(result.unwrap(), 2);

        // Should still have 1 and 3
        assert_eq!(consumer.pop().unwrap(), 1);
        assert_eq!(consumer.pop().unwrap(), 3);
    }

    #[test]
    fn test_consume_function() {
        let (producer, consumer) = queue_pack::<u32, 2, 4>()
            .queue_capacity(8)
            .channels()
            .unwrap();

        // Add some data
        for i in 0..5 {
            producer.push(i).unwrap();
        }

        let mut collected = Vec::new();
        let consumed = consumer.consume(|value, _seq| {
            collected.push(value);
            value >= 2 // Stop after finding value >= 2
        });

        assert!(consumed > 0);
        assert!(!collected.is_empty());
        assert!(collected.contains(&2) || collected.iter().any(|&x| x > 2));
    }
}
