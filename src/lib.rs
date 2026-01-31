//! # Jacques: High-Performance Lock-Free MPMC Queues
//!
//! Jacques is a high-performance, lock-free Multi-Producer Multi-Consumer
//! (MPMC) queue library designed for concurrent applications requiring maximum
//! throughput and minimal latency.
//!
//! ## Features
//!
//! - **Lock-free algorithms**: Zero mutex contention with atomic operations
//! - **MPMC support**: Multiple producers and consumers can operate
//!   concurrently
//! - **Zero-allocation operation**: No dynamic allocation during push/pop
//!   operations
//! - **Horizontal scaling**: Pack-based load distribution across multiple
//!   queues
//! - **Type safety**: Comprehensive compile-time guarantees with generic design
//! - **Memory efficient**: Packed 128-bit atomic operations with sequence
//!   numbers
//! - **Rich API**: Blocking, non-blocking, conditional, and bulk operations
//!
//! ## Queue Types
//!
//! Jacques provides three main queue implementations:
//!
//! ### 1. Owned Queue (`MpmcQueue`)
//! The foundational lock-free queue for `Copy` types:
//!
//! ```rust
//! use jacques::{
//!     owned::queue,
//!     traits::{QueueConsumer, QueueProducer},
//! };
//!
//! # #[tokio::main]
//! # async fn main() -> Result<(), jacques::QueueError> {
//! let (producer, consumer) = queue::<u64>().capacity(1024).channels()?;
//!
//! producer.push(42)?;
//! assert_eq!(consumer.pop()?, 42);
//! # Ok(())
//! # }
//! ```
//!
//! ### 2. Pointer Queue (`PointerQueue`)
//! Store non-Copy types by wrapping them in `Arc<T>`:
//!
//! ```rust
//! use jacques::pointer::pointer_queue;
//! use std::sync::Arc;
//!
//! #[derive(Debug, Clone, PartialEq)]
//! struct Message {
//!     id: u64,
//!     data: Vec<u8>,
//! }
//!
//! # #[tokio::main]
//! # async fn main() -> Result<(), jacques::QueueError> {
//! use jacques::traits::{QueueConsumer, QueueProducer};
//! let (producer, consumer) = pointer_queue::<Message>().capacity(512).channels()?;
//!
//! let msg = Arc::new(Message {
//!     id: 1,
//!     data: vec![1, 2, 3],
//! });
//! producer.push(msg.clone())?;
//! assert_eq!(consumer.pop()?, msg);
//! # Ok(())
//! # }
//! ```
//!
//! ### 3. Queue Pack (`QueuePack`)
//! Horizontal scaling with multiple independent queues:
//!
//! ```rust
//! use jacques::pack::queue_pack;
//!
//! # #[tokio::main]
//! # async fn main() -> Result<(), jacques::QueueError> {
//! // 4 queues, scan every 16 operations
//! use jacques::traits::{QueueConsumer, QueueProducer};
//! let (producer, consumer) = queue_pack::<u64, 4, 16>().queue_capacity(256).channels()?;
//!
//! producer.push(100)?;
//! assert_eq!(consumer.pop()?, 100);
//! # Ok(())
//! # }
//! ```
//!
//! ## Performance Characteristics
//!
//! - **Throughput**: >100M operations/second on modern hardware
//! - **Latency**: Sub-microsecond operation latency
//! - **Scalability**: Linear scaling with core count using queue packs
//! - **Memory**: Constant memory usage, no dynamic allocation
//!
//! ## Advanced Features
//!
//! ### Sequence Numbers
//! Track operation ordering across concurrent access:
//!
//! ```rust
//! use jacques::owned::queue;
//!
//! # #[tokio::main]
//! # async fn main() -> Result<(), jacques::QueueError> {
//! use jacques::traits::{QueueConsumer, QueueProducer};
//! let (producer, consumer) = queue::<u8>().capacity(64).channels()?;
//!
//! let seq = producer.push_with_seq(12)?;
//! let (value, pop_seq) = consumer.pop_with_seq()?;
//! assert_eq!(value, 12);
//! # Ok(())
//! # }
//! ```
//!
//! ### Conditional Operations
//! Process elements based on predicates:
//!
//! ```rust
//! use jacques::owned::queue;
//!
//! # #[tokio::main]
//! # async fn main() -> Result<(), jacques::QueueError> {
//! use jacques::traits::{QueueConsumer, QueueProducer};
//! let (producer, consumer) = queue::<i32>().capacity(32).channels()?;
//!
//! producer.push(2)?;
//! producer.push(1)?;
//! producer.push(3)?;
//!
//! // Pop the head if it's even
//! let even = consumer.pop_if(|&value, _seq| value % 2 == 0)?;
//! assert_eq!(even, 2);
//! # Ok(())
//! # }
//! ```
//!
//! ### Bulk Processing
//! Consume multiple elements efficiently:
//!
//! ```rust
//! use jacques::owned::queue;
//!
//! # #[tokio::main]
//! # async fn main() -> Result<(), jacques::QueueError> {
//! use jacques::traits::{QueueConsumer, QueueProducer};
//! let (producer, consumer) = queue::<u32>().capacity(16).channels()?;
//!
//! for i in 0..5 {
//!     producer.push(i)?;
//! }
//!
//! let mut sum = 0;
//! let count = consumer.consume(|value, _seq| {
//!     sum += value;
//!     value >= 3 // Stop after processing value 3
//! });
//!
//! println!("Processed {} items, sum: {}", count, sum);
//! # Ok(())
//! # }
//! ```
//!
//! ## Thread Safety
//!
//! All queue types are `Send + Sync` and designed for concurrent access:
//!
//! ```rust
//! use jacques::owned::queue;
//! use std::thread;
//!
//! # #[tokio::main]
//! # async fn main() -> Result<(), jacques::QueueError> {
//! use jacques::traits::{QueueConsumer, QueueProducer};
//! let (producer, consumer) = queue::<usize>().capacity(1024).channels()?;
//!
//! // Spawn producer thread
//! let producer_handle = {
//!     let producer = producer.clone();
//!     thread::spawn(move || {
//!         for i in 0..100 {
//!             producer.push(i).unwrap();
//!         }
//!     })
//! };
//!
//! // Spawn consumer thread
//! let consumer_handle = {
//!     let consumer = consumer.clone();
//!     thread::spawn(move || {
//!         let mut sum = 0;
//!         for _ in 0..100 {
//!             sum += consumer.pop().unwrap();
//!         }
//!         sum
//!     })
//! };
//!
//! producer_handle.join().unwrap();
//! let sum = consumer_handle.join().unwrap();
//! println!("Sum: {}", sum);
//! # Ok(())
//! # }
//! ```
//!
//! ## Memory Layout
//!
//! Jacques uses a carefully designed memory layout for optimal performance:
//! - 128-bit atomic operations containing both data and sequence numbers
//! - Cache-padded storage to prevent false sharing
//! - Power-of-two capacities for efficient modulo operations
//!
//! ## Error Handling
//!
//! All operations return `Result` types with descriptive errors:
//! - `QueueError::Full` - Queue capacity exceeded
//! - `QueueError::Empty` - No elements available
//! - `QueueError::InvalidCapacity` - Invalid configuration
//! - `QueueError::TypeSizeExceeded` - Type too large for atomic storage
//!
//! ## Minimum Supported Rust Version (MSRV)
//!
//! Jacques requires Rust 1.88 or later.
#![deny(
    missing_docs,
    unused_imports,
    unused_variables,
    dead_code,
    unreachable_code,
    unused_must_use
)]
#![warn(clippy::all, clippy::pedantic, clippy::nursery, clippy::cargo)]
#![allow(
    clippy::module_name_repetitions,
    clippy::similar_names,
    clippy::too_many_arguments,
    clippy::must_use_candidate,
    clippy::missing_errors_doc,
    clippy::cast_precision_loss,
    clippy::type_complexity,
    clippy::similar_names
)]
#![cfg_attr(docsrs, feature(doc_cfg))]

/// Core lock-free MPMC queue implementation for `Copy` types.
///
/// This module provides the foundational [`MpmcQueue`] implementation and
/// associated builder patterns, producer/consumer handles, and convenience
/// functions.
///
/// [`MpmcQueue`]: owned::MpmcQueue
pub mod owned;

/// Horizontal scaling with multiple independent queues.
///
/// This module provides [`QueuePack`] which distributes load across multiple
/// independent queues for better performance and reduced contention in
/// high-throughput scenarios.
///
/// [`QueuePack`]: pack::QueuePack
pub mod pack;

/// Lock-free MPMC queue for non-`Copy` types using `Arc<T>` storage.
///
/// This module provides [`PointerQueue`] which enables storing arbitrary types
/// by converting `Arc<T>` values to raw pointers internally, maintaining the
/// performance characteristics of the underlying copy-based queue.
///
/// [`PointerQueue`]: pointer::PointerQueue
pub mod pointer;

/// Common traits for queue producers, consumers, and factories.
///
/// This module defines the core abstractions that enable consistent APIs across
/// all queue implementations: [`QueueProducer`], [`QueueConsumer`], and
/// [`QueueFactory`].
///
/// [`QueueProducer`]: traits::QueueProducer
/// [`QueueConsumer`]: traits::QueueConsumer
/// [`QueueFactory`]: traits::QueueFactory
pub mod traits;

use std::{mem, mem::MaybeUninit, ptr};
use thiserror::Error;

/// Errors that can occur during queue operations.
///
/// This enum provides comprehensive error reporting for all queue operations,
/// enabling robust error handling in concurrent applications.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum QueueError {
    /// The queue has reached its maximum capacity and cannot accept more
    /// elements.
    ///
    /// This error occurs when attempting to push to a full queue using
    /// non-blocking operations like `try_push`. Blocking operations will
    /// spin until space becomes available.
    #[error("queue is full")]
    Full,

    /// The queue contains no elements to consume.
    ///
    /// This error occurs when attempting to pop from an empty queue using
    /// non-blocking operations like `try_pop`. Blocking operations will
    /// spin until elements become available.
    #[error("queue is empty")]
    Empty,

    /// The specified capacity is invalid.
    ///
    /// Queue capacities must be powers of two and at least 2. This constraint
    /// enables efficient bit-masking operations for index calculations.
    #[error("invalid capacity: must be a power of two and >= 2")]
    InvalidCapacity,

    /// The combined size of data and index types exceeds the 16-byte atomic
    /// storage limit.
    ///
    /// Jacques packs data and sequence numbers into 128-bit atomic operations.
    /// The total size of your data type plus the index type must not exceed
    /// 16 bytes.
    #[error("type size constraints violated (data + index must fit in 16 bytes)")]
    TypeSizeExceeded {
        /// The size of the data type that caused the constraint violation.
        size: usize,
    },

    /// The runtime capacity does not match the compile-time capacity for static
    /// queues.
    ///
    /// When using const generic capacity parameters, the runtime capacity
    /// parameter must match the compile-time parameter exactly.
    #[error("capacity mismatch for compile-time queue")]
    CapacityMismatch,
}

/// Packs data and sequence number into a single 128-bit atomic value.
///
/// This function implements the core memory layout for Jacques queues by
/// combining both data and sequence information into a single atomic operation.
/// The layout uses the upper bits for data and lower bits for the sequence
/// number.
///
/// # Memory Layout
///
/// ```text
/// |--- Data (128-seq_shift bits) ---|--- Sequence (seq_shift bits) ---|
/// |                                  |                                  |
/// u128 = (data_u128 << seq_shift) | (seq & seq_mask)
/// ```
///
/// # Safety
///
/// This function uses `unsafe` code for byte-level copying but maintains safety
/// by:
/// - Only copying `size_of::<T>()` bytes
/// - Using a properly sized 16-byte buffer
/// - Ensuring `T: Copy` for bitwise copying safety
///
/// # Parameters
///
/// - `seq`: The sequence number to pack (only lower `seq_shift` bits used)
/// - `data`: The data value to pack
/// - `seq_shift`: Number of bits allocated for sequence numbers
///
/// # Returns
///
/// A packed 128-bit value suitable for atomic storage
#[allow(clippy::extra_unused_type_parameters)]
const fn pack_entry<T, I>(seq: u128, data: T, seq_shift: u32) -> u128
where
    T: Copy,
    I: Copy,
{
    let data_size = mem::size_of::<T>();
    let mut buf = [0u8; 16];

    // SAFETY: We copy exactly `data_size` bytes from a valid `T` reference
    // into a 16-byte buffer, which is always sufficient since we validate
    // that `size_of::<T>() + size_of::<I>() <= 16` during queue creation.
    unsafe {
        ptr::copy_nonoverlapping((&raw const data).cast::<u8>(), buf.as_mut_ptr(), data_size);
    }

    let data_u128 = u128::from_le_bytes(buf);
    let seq_mask = (1u128 << seq_shift) - 1u128;
    (data_u128 << seq_shift) | (seq & seq_mask)
}

/// Unpacks data and sequence number from a 128-bit atomic value.
///
/// This function reverses the packing operation, extracting both the sequence
/// number and data from a single atomic value while maintaining memory safety.
///
/// # Safety
///
/// This function uses `unsafe` code for byte-level copying but maintains safety
/// by:
/// - Only copying the exact size of `T` as specified by `data_size`
/// - Using `MaybeUninit` to handle uninitialized memory properly
/// - Ensuring `T: Copy` for safe bitwise reconstruction
///
/// # Parameters
///
/// - `val`: The packed 128-bit value from atomic storage
/// - `seq_shift`: Number of bits allocated for sequence numbers
/// - `data_size`: Size in bytes of the data type `T`
///
/// # Returns
///
/// A tuple containing `(sequence_number, unpacked_data)`
#[allow(clippy::extra_unused_type_parameters)]
const fn unpack_entry<T, I>(val: u128, seq_shift: u32, data_size: usize) -> (u128, T)
where
    T: Copy,
    I: Copy,
{
    let seq_mask = (1u128 << seq_shift) - 1u128;
    let seq = val & seq_mask;
    let data_u128 = val >> seq_shift;
    let bytes = data_u128.to_le_bytes();

    let mut t_uninit = MaybeUninit::<T>::uninit();

    // SAFETY: We copy exactly `data_size` bytes (which equals `size_of::<T>()`)
    // from the byte array into uninitialized memory, then assume initialization.
    // This is safe because:
    // 1. `T: Copy` guarantees bitwise copying is valid
    // 2. We copy the exact number of bytes that make up `T`
    // 3. The bytes came from a valid `T` during packing
    unsafe {
        ptr::copy_nonoverlapping(
            bytes.as_ptr(),
            t_uninit.as_mut_ptr().cast::<u8>(),
            data_size,
        );
        (seq, t_uninit.assume_init())
    }
}
