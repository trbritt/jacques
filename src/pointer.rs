use crate::{
    QueueError,
    owned::{MpmcQueue, QueueBuilder},
    traits::{QueueConsumer, QueueFactory, QueueProducer},
};
use std::{fmt, marker::PhantomData, sync::Arc};

/// A lock-free MPMC queue that stores `Arc<T>` values by converting them to raw
/// pointers internally.
///
/// This queue enables storing non-`Copy` types while maintaining the
/// performance characteristics of the underlying copy-based queue. It achieves
/// this by:
/// 1. Converting `Arc<T>` to raw pointers (`usize`) for storage
/// 2. Storing raw pointers in the underlying `MpmcQueue<usize>`
/// 3. Reconstructing `Arc<T>` on retrieval
///
/// # Type Parameters
///
/// * `T` - The data type to store (must be `Send + Sync`)
/// * `I` - The index/sequence number type (default: `u32`)
/// * `N` - Compile-time capacity (0 = dynamic, >0 = static)
///
/// # Memory Management
///
/// The queue properly manages `Arc` reference counts:
/// - `push()` consumes an `Arc<T>` and converts it to a raw pointer
/// - `pop()` reconstructs the `Arc<T>` from the raw pointer
/// - `peek()` clones the `Arc<T>` without consuming it
///
/// # Safety
///
/// All pointer conversions are safe because:
/// - Raw pointers are only created from `Arc::into_raw()`
/// - Raw pointers are only converted back via `Arc::from_raw()`
/// - The queue ensures proper ownership transfer
///
/// # Examples
///
/// ```
/// use jacques::pointer::pointer_queue;
/// use std::sync::Arc;
///
/// #[derive(Debug, PartialEq)]
/// struct Data {
///     id: u64,
///     payload: Vec<u8>,
/// }
///
/// # fn main() -> Result<(), jacques::QueueError> {
/// use jacques::traits::{QueueConsumer, QueueProducer};
/// let (producer, consumer) = pointer_queue::<Data>().capacity(128).channels()?;
///
/// let data = Arc::new(Data {
///     id: 1,
///     payload: vec![1, 2, 3],
/// });
///
/// producer.push(data.clone())?;
/// assert_eq!(consumer.pop()?, data);
/// # Ok(())
/// # }
/// ```
pub struct PointerQueue<T, I = u32, const N: usize = 0>
where
    T: Send + Sync,
    I: Copy + Into<u128>,
{
    queue: Arc<MpmcQueue<usize, I, N>>,
    _phantom: PhantomData<T>,
}

impl<T, I, const N: usize> fmt::Debug for PointerQueue<T, I, N>
where
    T: Send + Sync + fmt::Debug,
    I: Copy + Into<u128> + fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PointerQueue")
            .field("capacity", &self.capacity())
            .field("len", &self.len())
            .field("is_empty", &self.is_empty())
            .finish()
    }
}

/// Builder for pointer queues.
///
/// Provides a fluent API for constructing pointer queues with validated
/// parameters. Wraps the underlying `QueueBuilder<usize>` to provide type-safe
/// construction of queues that store `Arc<T>`.
///
/// # Type Parameters
///
/// * `T` - The data type to store (must be `Send + Sync`)
/// * `I` - The index/sequence number type (default: `u32`)
///
/// # Examples
///
/// ```
/// use jacques::pointer::pointer_queue;
/// use std::sync::Arc;
///
/// #[derive(Clone)]
/// struct Message {
///     text: String,
/// }
///
/// # fn main() -> Result<(), jacques::QueueError> {
/// use jacques::traits::{QueueConsumer, QueueProducer};
/// let (producer, consumer) = pointer_queue::<Message>().capacity(256).channels()?;
///
/// let msg = Arc::new(Message {
///     text: "Hello".to_string(),
/// });
/// producer.push(msg)?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct PointerQueueBuilder<T, I = u32>
where
    T: Send + Sync,
    I: Copy + Into<u128>,
{
    inner: QueueBuilder<usize, I>,
    _phantom: PhantomData<T>,
}

impl<T, I> Default for PointerQueueBuilder<T, I>
where
    T: Send + Sync,
    I: Copy + Into<u128>,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T, I> PointerQueueBuilder<T, I>
where
    T: Send + Sync,
    I: Copy + Into<u128>,
{
    /// Create a new pointer queue builder
    pub const fn new() -> Self {
        Self {
            inner: QueueBuilder::new(),
            _phantom: PhantomData,
        }
    }

    /// Set the queue capacity (must be a power of 2)
    #[must_use]
    pub const fn capacity(mut self, cap: usize) -> Self {
        self.inner = self.inner.capacity(cap);
        self
    }

    /// Build a dynamic pointer queue
    pub fn build(self) -> Result<Arc<PointerQueue<T, I>>, QueueError> {
        let queue = self.inner.build()?;
        Ok(Arc::new(PointerQueue {
            queue,
            _phantom: PhantomData,
        }))
    }

    /// Build a static pointer queue with compile-time capacity
    pub fn build_static<const N: usize>(self) -> Result<Arc<PointerQueue<T, I, N>>, QueueError> {
        let queue = self.inner.build_static::<N>()?;
        Ok(Arc::new(PointerQueue {
            queue,
            _phantom: PhantomData,
        }))
    }

    /// Create producer/consumer pair
    pub fn channels(self) -> Result<(PointerProducer<T, I>, PointerConsumer<T, I>), QueueError> {
        let queue = self.build()?;
        Ok((queue.producer(), queue.consumer()))
    }

    /// Create producer/consumer pair with static capacity
    pub fn channels_static<const N: usize>(
        self,
    ) -> Result<(PointerProducer<T, I, N>, PointerConsumer<T, I, N>), QueueError> {
        let queue = self.build_static::<N>()?;
        Ok((queue.producer(), queue.consumer()))
    }
}

/// Convenience function for creating pointer queues with default index type
/// (`u32`).
///
/// This is the primary entry point for creating pointer-based MPMC queues.
/// Returns a builder that allows configuring capacity and other parameters.
///
/// # Examples
///
/// ```
/// use jacques::pointer::pointer_queue;
/// use std::sync::Arc;
///
/// #[derive(Debug, PartialEq)]
/// struct Task {
///     id: usize,
///     data: Vec<u8>,
/// }
///
/// # fn main() -> Result<(), jacques::QueueError> {
/// use jacques::traits::{QueueConsumer, QueueProducer};
/// let (producer, consumer) = pointer_queue::<Task>().capacity(512).channels()?;
///
/// let task = Arc::new(Task {
///     id: 1,
///     data: vec![0; 100],
/// });
/// producer.push(task.clone())?;
/// assert_eq!(consumer.pop()?, task);
/// # Ok(())
/// # }
/// ```
pub const fn pointer_queue<T>() -> PointerQueueBuilder<T, u32>
where
    T: Send + Sync,
{
    PointerQueueBuilder::new()
}

/// Convenience function for creating pointer queues with custom index type.
///
/// Use this when you need larger sequence numbers or want to optimize memory
/// usage.
///
/// # Type Parameters
///
/// * `T` - The data type to store
/// * `I` - The index/sequence number type
///
/// # Examples
///
/// ```
/// use jacques::pointer::pointer_queue_with_index;
/// use std::sync::Arc;
///
/// struct LargeData(Vec<u8>);
///
/// # fn main() -> Result<(), jacques::QueueError> {
/// use jacques::traits::QueueProducer;
/// // Use u64 for larger sequence numbers
/// let (producer, _consumer) = pointer_queue_with_index::<LargeData, u64>()
///     .capacity(1024)
///     .channels()?;
///
/// producer.push(Arc::new(LargeData(vec![0; 1000])))?;
/// # Ok(())
/// # }
/// ```
pub const fn pointer_queue_with_index<T, I>() -> PointerQueueBuilder<T, I>
where
    T: Send + Sync,
    I: Copy + Into<u128>,
{
    PointerQueueBuilder::new()
}

impl<T, I, const N: usize> PointerQueue<T, I, N>
where
    T: Send + Sync,
    I: Copy + Into<u128>,
{
    /// Create a new pointer queue with the specified capacity
    pub fn new(capacity: usize) -> Result<Self, QueueError> {
        let queue = Arc::new(MpmcQueue::new(capacity)?);
        Ok(Self {
            queue,
            _phantom: PhantomData,
        })
    }

    /// Get the capacity of the queue
    pub fn capacity(&self) -> usize {
        self.queue.capacity()
    }

    /// Get the current number of elements in the queue
    pub fn len(&self) -> usize {
        self.queue.len()
    }

    /// Check if the queue is empty
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    /// Check if the queue is full
    pub fn is_full(&self) -> bool {
        self.queue.is_full()
    }

    /// Try to push an Arc without blocking
    pub fn try_push(&self, arc: Arc<T>) -> Result<(), (Arc<T>, QueueError)> {
        let raw_ptr = Arc::into_raw(arc) as usize;

        match self.queue.try_push(raw_ptr) {
            Ok(()) => Ok(()),
            Err((ptr, err)) => {
                // Reconstruct the Arc to avoid leaking
                let recovered_arc = unsafe { Arc::from_raw(ptr as *const T) };
                Err((recovered_arc, err))
            },
        }
    }

    /// Push an Arc, spinning until successful
    pub fn push(&self, arc: Arc<T>) -> Result<(), QueueError> {
        let raw_ptr = Arc::into_raw(arc) as usize;
        self.queue.push(raw_ptr)
    }

    /// Try to pop an Arc without blocking
    pub fn try_pop(&self) -> Result<Arc<T>, QueueError> {
        match self.queue.try_pop() {
            Ok(raw_ptr) => {
                let arc = unsafe { Arc::from_raw(raw_ptr as *const T) };
                Ok(arc)
            },
            Err(e) => Err(e),
        }
    }

    /// Pop an Arc, spinning until successful
    pub fn pop(&self) -> Result<Arc<T>, QueueError> {
        match self.queue.pop() {
            Ok(raw_ptr) => {
                let arc = unsafe { Arc::from_raw(raw_ptr as *const T) };
                Ok(arc)
            },
            Err(e) => Err(e),
        }
    }

    /// Peek at the front Arc without removing it
    pub fn peek(&self) -> Result<Arc<T>, QueueError> {
        match self.queue.peek() {
            Ok(raw_ptr) => {
                // Safely peek by temporarily reconstructing the Arc and cloning it
                let temp_arc = unsafe { Arc::from_raw(raw_ptr as *const T) };
                let cloned_arc = Arc::clone(&temp_arc);
                // Leak the temporary Arc back (don't drop it)
                let _ = Arc::into_raw(temp_arc);
                Ok(cloned_arc)
            },
            Err(e) => Err(e),
        }
    }
}

// Type aliases for common configurations

/// Convenient type alias for [`PointerProducerHandle`].
///
/// This simplifies the type signatures when using pointer queue producer
/// handles with default configuration parameters.
pub type PointerProducer<T, I = u32, const N: usize = 0> = PointerProducerHandle<T, I, N>;

/// Convenient type alias for [`PointerConsumerHandle`].
///
/// This simplifies the type signatures when using pointer queue consumer
/// handles with default configuration parameters.
pub type PointerConsumer<T, I = u32, const N: usize = 0> = PointerConsumerHandle<T, I, N>;

/// Producer handle for the pointer queue.
///
/// A lightweight, cloneable handle that allows pushing `Arc<T>` items to the
/// queue. Multiple producer handles can be created for the same queue, enabling
/// multi-producer scenarios.
///
/// # Type Parameters
///
/// * `T` - The data type to store (must be `Send + Sync`)
/// * `I` - The index/sequence number type (default: `u32`)
/// * `N` - Compile-time capacity (0 = dynamic, >0 = static)
///
/// # Examples
///
/// ```
/// use jacques::{pointer::pointer_queue, traits::QueueProducer};
/// use std::{sync::Arc, thread};
///
/// struct Event {
///     id: u64,
///     message: String,
/// }
///
/// # fn main() -> Result<(), jacques::QueueError> {
/// let (producer, consumer) = pointer_queue::<Event>().capacity(256).channels()?;
///
/// // Clone producer for another thread
/// let producer2 = producer.clone();
/// let handle = thread::spawn(move || {
///     let event = Arc::new(Event {
///         id: 1,
///         message: "Hello".to_string(),
///     });
///     producer2.push(event).unwrap();
/// });
///
/// handle.join().unwrap();
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct PointerProducerHandle<T, I = u32, const N: usize = 0>
where
    T: Send + Sync,
    I: Copy + Into<u128>,
{
    queue: Arc<PointerQueue<T, I, N>>,
}

impl<T, I, const N: usize> Clone for PointerProducerHandle<T, I, N>
where
    T: Send + Sync,
    I: Copy + Into<u128>,
{
    fn clone(&self) -> Self {
        Self {
            queue: self.queue.clone(),
        }
    }
}

impl<T, I, const N: usize> QueueProducer<Arc<T>> for PointerProducerHandle<T, I, N>
where
    T: Send + Sync,
    I: Copy + Into<u128>,
{
    fn try_push(&self, value: Arc<T>) -> Result<(), (Arc<T>, QueueError)> {
        self.queue.try_push(value)
    }
    fn push(&self, arc: Arc<T>) -> Result<(), QueueError> {
        self.queue.push(arc)
    }

    fn push_with_seq(&self, arc: Arc<T>) -> Result<usize, QueueError> {
        let raw_ptr = Arc::into_raw(arc) as usize;
        self.queue.queue.push_impl(raw_ptr, true)
    }
}

/// Consumer handle for the pointer queue.
///
/// A lightweight, cloneable handle that allows popping `Arc<T>` items from the
/// queue. Multiple consumer handles can be created for the same queue, enabling
/// multi-consumer scenarios.
///
/// Provides rich consumption APIs including conditional popping, bulk
/// consumption, and peeking without removal. All operations properly manage
/// `Arc` reference counts.
///
/// # Type Parameters
///
/// * `T` - The data type to store (must be `Send + Sync`)
/// * `I` - The index/sequence number type (default: `u32`)
/// * `N` - Compile-time capacity (0 = dynamic, >0 = static)
///
/// # Examples
///
/// ```
/// use jacques::{
///     pointer::pointer_queue,
///     traits::{QueueConsumer, QueueProducer},
/// };
/// use std::sync::Arc;
///
/// #[derive(Debug, PartialEq)]
/// struct Job {
///     id: u64,
///     priority: u8,
/// }
///
/// # fn main() -> Result<(), jacques::QueueError> {
/// let (producer, consumer) = pointer_queue::<Job>().capacity(128).channels()?;
///
/// producer.push(Arc::new(Job { id: 1, priority: 1 }))?;
/// producer.push(Arc::new(Job { id: 2, priority: 5 }))?;
/// producer.push(Arc::new(Job { id: 3, priority: 3 }))?;
///
/// // Conditional pop - only high priority jobs
/// if let Ok(job) = consumer.pop_if(|job, _| job.priority >= 5) {
///     assert_eq!(job.id, 2);
/// }
///
/// // Peek without consuming
/// let peeked = consumer.peek()?;
/// assert_eq!(peeked.id, 1);
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct PointerConsumerHandle<T, I = u32, const N: usize = 0>
where
    T: Send + Sync,
    I: Copy + Into<u128>,
{
    queue: Arc<PointerQueue<T, I, N>>,
}

impl<T, I, const N: usize> Clone for PointerConsumerHandle<T, I, N>
where
    T: Send + Sync,
    I: Copy + Into<u128>,
{
    fn clone(&self) -> Self {
        Self {
            queue: self.queue.clone(),
        }
    }
}

impl<T, I, const N: usize> QueueConsumer<Arc<T>> for PointerConsumerHandle<T, I, N>
where
    T: Send + Sync,
    I: Copy + Into<u128>,
{
    fn try_pop(&self) -> Result<Arc<T>, QueueError> {
        self.queue.try_pop()
    }
    fn pop(&self) -> Result<Arc<T>, QueueError> {
        self.queue.pop()
    }

    fn pop_with_seq(&self) -> Result<(Arc<T>, usize), QueueError> {
        match self.queue.queue.pop_impl(true) {
            Ok((raw_ptr, idx)) => {
                let arc = unsafe { Arc::from_raw(raw_ptr as *const T) };
                Ok((arc, idx))
            },
            Err(e) => Err(e),
        }
    }

    fn peek(&self) -> Result<Arc<T>, QueueError> {
        self.queue.peek()
    }

    fn peek_with_seq(&self) -> Result<(Arc<T>, usize), QueueError> {
        // TODO(fickle approximation)
        let seq = self.queue.queue.len();
        let arc = self.queue.peek()?;
        Ok((arc, seq))
    }

    fn pop_if<F>(&self, mut predicate: F) -> Result<Arc<T>, QueueError>
    where
        F: FnMut(&Arc<T>, usize) -> bool,
    {
        // Peek first to check predicate
        let (peeked_arc, seq) = self.peek_with_seq()?;

        if predicate(&peeked_arc, seq) {
            // Race condition possible here
            self.pop()
        } else {
            Err(QueueError::Empty)
        }
    }

    fn consume<F>(&self, mut consumer: F) -> usize
    where
        F: FnMut(Arc<T>, usize) -> bool,
    {
        let mut count = 0;
        while let Ok((arc, seq)) = self.pop_with_seq() {
            count += 1;
            if consumer(arc, seq) {
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

impl<T, I, const N: usize> QueueFactory<Arc<T>> for Arc<PointerQueue<T, I, N>>
where
    T: Send + Sync,
    I: Copy + Into<u128>,
{
    type Producer = PointerProducerHandle<T, I, N>;
    type Consumer = PointerConsumerHandle<T, I, N>;

    fn producer(&self) -> Self::Producer {
        PointerProducerHandle {
            queue: self.clone(),
        }
    }

    fn consumer(&self) -> Self::Consumer {
        PointerConsumerHandle {
            queue: self.clone(),
        }
    }
}

// Safety: The queue only stores raw pointers derived from Arc<T> and manages
// ownership correctly
unsafe impl<T, I, const N: usize> Send for PointerQueue<T, I, N>
where
    T: Send + Sync,
    I: Copy + Into<u128>,
{
}

unsafe impl<T, I, const N: usize> Sync for PointerQueue<T, I, N>
where
    T: Send + Sync,
    I: Copy + Into<u128>,
{
}

impl<T, I, const N: usize> Drop for PointerQueue<T, I, N>
where
    T: Send + Sync,
    I: Copy + Into<u128>,
{
    fn drop(&mut self) {
        // Clean up remaining items
        while self.try_pop().is_ok() {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        sync::atomic::{AtomicUsize, Ordering},
        time::Instant,
    };

    #[derive(Debug, Clone, PartialEq)]
    struct LargeData {
        id: u64,
        data: Vec<u8>,
        name: String,
    }

    impl LargeData {
        fn new(id: u64, size: usize) -> Self {
            Self {
                id,
                data: vec![0u8; size],
                name: format!("item_{id}"),
            }
        }
    }

    #[test]
    fn test_builder_pattern() {
        let queue = pointer_queue::<LargeData>().capacity(16).build().unwrap();

        assert_eq!(queue.capacity(), 16);
        assert!(queue.is_empty());
    }

    #[test]
    fn test_channels() {
        let (producer, consumer) = pointer_queue::<LargeData>().capacity(8).channels().unwrap();

        let data1 = Arc::new(LargeData::new(1, 1024));
        let data2 = Arc::new(LargeData::new(2, 2048));

        producer.push(data1.clone()).unwrap();
        producer.push(data2.clone()).unwrap();

        let popped1 = consumer.pop().unwrap();
        let popped2 = consumer.pop().unwrap();

        assert_eq!(*popped1, *data1);
        assert_eq!(*popped2, *data2);
        assert!(consumer.is_empty());
    }

    #[test]
    fn test_try_operations() {
        let queue = pointer_queue::<LargeData>().capacity(2).build().unwrap();

        let data1 = Arc::new(LargeData::new(1, 128));
        let data2 = Arc::new(LargeData::new(2, 128));
        let data3 = Arc::new(LargeData::new(3, 128));

        // Fill the queue
        assert!(queue.try_push(data1.clone()).is_ok());
        assert!(queue.try_push(data2).is_ok());

        // Queue should be full
        assert!(queue.is_full());

        // Next push should fail and return the Arc
        match queue.try_push(data3.clone()) {
            Err((returned_arc, QueueError::Full)) => {
                assert_eq!(*returned_arc, *data3);
            },
            _ => panic!("Expected full queue error"),
        }

        // Pop should work
        let popped = queue.try_pop().unwrap();
        assert_eq!(*popped, *data1);
    }

    #[test]
    fn test_peek() {
        let queue = pointer_queue::<LargeData>().capacity(4).build().unwrap();

        let data = Arc::new(LargeData::new(99, 256));
        queue.push(data.clone()).unwrap();

        // Peek should return a clone without consuming
        let peeked = queue.peek().unwrap();
        assert_eq!(*peeked, *data);

        // Original should still be in queue
        let popped = queue.pop().unwrap();
        assert_eq!(*popped, *data);
    }

    #[test]
    fn test_reference_counting() {
        let queue = pointer_queue::<LargeData>().capacity(4).build().unwrap();

        let data = Arc::new(LargeData::new(123, 64));

        // Initially, we have 1 reference
        assert_eq!(Arc::strong_count(&data), 1);

        queue.push(data.clone()).unwrap();

        // After push, queue owns one reference, we still have one
        assert_eq!(Arc::strong_count(&data), 2);

        let peeked = queue.peek().unwrap();

        // After peek, we have: original + queue + peeked = 3
        assert_eq!(Arc::strong_count(&data), 3);

        drop(peeked);

        // Back to 2 after dropping peeked
        assert_eq!(Arc::strong_count(&data), 2);

        let popped = queue.pop().unwrap();

        // Still 2: original + popped
        assert_eq!(Arc::strong_count(&data), 2);

        drop(popped);

        // Back to 1: just original
        assert_eq!(Arc::strong_count(&data), 1);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn stress_test() {
        const CAPACITY: usize = 128;
        const PRODUCERS: usize = 2;
        const CONSUMERS: usize = 2;
        const ITEMS_PER_PRODUCER: usize = 10_000;

        let (producer, consumer) = pointer_queue::<LargeData>()
            .capacity(CAPACITY)
            .channels()
            .unwrap();

        let total_items = PRODUCERS * ITEMS_PER_PRODUCER;
        let consumed_count = Arc::new(AtomicUsize::new(0));

        // Spawn consumers
        let mut consumer_handles = Vec::new();
        for _ in 0..CONSUMERS {
            let consumer = consumer.clone();
            let consumed_clone = consumed_count.clone();

            let handle = tokio::task::spawn(async move {
                loop {
                    if consumed_clone.load(Ordering::SeqCst) >= total_items {
                        break;
                    }

                    match consumer.try_pop() {
                        Ok(_data) => {
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
                    let data = Arc::new(LargeData::new(
                        (producer_id * ITEMS_PER_PRODUCER + item_id) as u64,
                        64,
                    ));

                    loop {
                        match producer.try_push(data.clone()) {
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

        // Wait for all producers
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
            "Pointer queue stress test: {PRODUCERS} producers, {CONSUMERS} consumers, {ITEMS_PER_PRODUCER} items each = {total_items} total in {elapsed:?} ({throughput:.0} ops/sec)"
        );

        assert_eq!(consumed_count.load(Ordering::SeqCst), total_items);
    }
}
