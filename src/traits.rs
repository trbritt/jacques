use crate::QueueError;

/// Trait for queue producers that can push items into a queue.
///
/// This trait provides a consistent interface for all producer types,
/// whether they're direct queue references, dedicated producer handles,
/// or pack-based producers.
pub trait QueueProducer<T> {
    /// Push a value into the queue but does not block if we cannot push
    /// it to the queue.
    ///
    /// # Arguments
    /// * `value` - The value to push
    ///
    /// # Returns
    /// `Ok(())` on success, or `QueueError` if the operation fails
    fn try_push(&self, value: T) -> Result<(), (T, QueueError)>;
    /// Push a value into the queue.
    ///
    /// # Arguments
    /// * `value` - The value to push
    ///
    /// # Returns
    /// `Ok(())` on success, or `QueueError` if the operation fails
    fn push(&self, value: T) -> Result<(), QueueError>;

    /// Push a value and get the sequence number.
    ///
    /// # Arguments
    /// * `value` - The value to push
    ///
    /// # Returns
    /// The sequence number on success, or `QueueError` if the operation fails
    fn push_with_seq(&self, value: T) -> Result<usize, QueueError>;
}

/// Trait for queue consumers that can pop items from a queue.
///
/// This trait provides a consistent interface for all consumer types,
/// whether they're direct queue references, dedicated consumer handles,
/// or pack-based consumers.
pub trait QueueConsumer<T> {
    /// Pop a value from the queue, but does not block if we cannot
    /// immediately pop a value.
    ///
    /// # Returns
    /// The popped value on success, or `QueueError::Empty` if the queue is
    /// empty
    fn try_pop(&self) -> Result<T, QueueError>;
    /// Pop a value from the queue.
    ///
    /// # Returns
    /// The popped value on success, or `QueueError::Empty` if the queue is
    /// empty
    fn pop(&self) -> Result<T, QueueError>;

    /// Pop a value with sequence number from the queue.
    ///
    /// # Returns
    /// The popped value and sequence number on success, or `QueueError::Empty`
    /// if empty
    fn pop_with_seq(&self) -> Result<(T, usize), QueueError>;

    /// Peek at the head element without removing it.
    ///
    /// # Returns
    /// A copy/clone of the head element, or `QueueError::Empty` if the queue is
    /// empty
    fn peek(&self) -> Result<T, QueueError>;

    /// Peek at the head element with sequence number.
    ///
    /// # Returns
    /// The head element and sequence number, or `QueueError::Empty` if empty
    fn peek_with_seq(&self) -> Result<(T, usize), QueueError>;

    /// Pop if predicate returns true for the head element.
    ///
    /// # Arguments
    /// * `predicate` - Function to test the head element
    ///
    /// # Returns
    /// The popped value if predicate succeeded, or `QueueError` if failed or
    /// empty
    fn pop_if<F>(&self, predicate: F) -> Result<T, QueueError>
    where
        F: FnMut(&T, usize) -> bool;

    /// Consume elements with a closure until queue is empty or closure returns
    /// true to stop.
    ///
    /// # Arguments
    /// * `consumer` - Function to process each element, returns true to stop
    ///
    /// # Returns
    /// Number of elements consumed
    fn consume<F>(&self, consumer: F) -> usize
    where
        F: FnMut(T, usize) -> bool;

    /// Check if the queue appears empty.
    /// Note: In concurrent scenarios, this may race with other operations.
    ///
    /// # Returns
    /// `true` if the queue appears empty
    fn is_empty(&self) -> bool;

    /// Get approximate queue size.
    /// Note: In concurrent scenarios, this may not be exact.
    ///
    /// # Returns
    /// Approximate number of elements in the queue
    fn size(&self) -> usize;
}

/// Trait for queues that can create producers and consumers.
///
/// This extends the basic `Queue` trait to provide a consistent API
/// for obtaining producer and consumer handles.
pub trait QueueFactory<T> {
    /// The type of producers this queue creates
    type Producer: QueueProducer<T>;

    /// The type of consumers this queue creates
    type Consumer: QueueConsumer<T>;

    /// Create both producer and consumer handles in one call.
    ///
    /// This is a convenience method equivalent to calling both `producer()` and
    /// `consumer()`.
    ///
    /// # Returns
    /// A tuple containing `(producer, consumer)` handles
    fn channel(&self) -> (Self::Producer, Self::Consumer) {
        (self.producer(), self.consumer())
    }

    /// Create a new producer handle for this queue.
    ///
    /// # Returns
    /// A producer that can push items to this queue
    fn producer(&self) -> Self::Producer;

    /// Create a new consumer handle for this queue.
    ///
    /// # Returns
    /// A consumer that can pop items from this queue
    fn consumer(&self) -> Self::Consumer;
}
