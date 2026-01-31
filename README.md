# Jacques

High-performance, lock-free Multi-Producer Multi-Consumer (MPMC) queue library designed for concurrent applications requiring maximum throughput and minimal latency.
Based on the queue implementation of [Erez Strauss](https://github.com/erez-strauss/lockfree_mpmc_queue/tree/master).
## Features

- **Lock-free algorithms**: Zero mutex contention with atomic operations
- **MPMC support**: Multiple producers and consumers can operate concurrently
- **Zero-allocation operation**: No dynamic allocation during push/pop operations
- **Horizontal scaling**: Pack-based load distribution across multiple queues
- **Type safety**: Comprehensive compile-time guarantees with generic design
- **Memory efficient**: Packed 128-bit atomic operations with sequence numbers
- **Rich API**: Blocking, non-blocking, conditional, and bulk operations

## Performance Characteristics

- **Throughput**: >100M operations/second on modern hardware
- **Latency**: Sub-microsecond operation latency
- **Scalability**: Linear scaling with core count using queue packs
- **Memory**: Constant memory usage, no dynamic allocation

## Queue Types

### Owned Queue (`MpmcQueue`)

The foundational lock-free queue for `Copy` types:

```rust
use jacques::{
    owned::queue,
    traits::{QueueConsumer, QueueProducer},
};

let (producer, consumer) = queue::<u64>().capacity(1024).channels()?;

producer.push(42)?;
assert_eq!(consumer.pop()?, 42);
```

### Pointer Queue (`PointerQueue`)

Store non-Copy types by wrapping them in `Arc<T>`:

```rust
use jacques::pointer::pointer_queue;
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq)]
struct Message {
    id: u64,
    data: Vec<u8>,
}

use jacques::traits::{QueueConsumer, QueueProducer};
let (producer, consumer) = pointer_queue::<Message>().capacity(512).channels()?;

let msg = Arc::new(Message {
    id: 1,
    data: vec![1, 2, 3],
});
producer.push(msg.clone())?;
assert_eq!(consumer.pop()?, msg);
```

### Queue Pack (`QueuePack`)

Horizontal scaling with multiple independent queues:

```rust
use jacques::pack::queue_pack;
use jacques::traits::{QueueConsumer, QueueProducer};

// 4 queues, scan every 16 operations
let (producer, consumer) = queue_pack::<u64, 4, 16>().queue_capacity(256).channels()?;

producer.push(100)?;
assert_eq!(consumer.pop()?, 100);
```

## Advanced Features

- **Sequence Numbers**: Track operation ordering across concurrent access
- **Conditional Operations**: Process elements based on predicates
- **Bulk Processing**: Consume multiple elements efficiently
- **Thread Safety**: All queue types are `Send + Sync`

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
jacques = "0.1"
```

## Documentation

For full documentation, visit [docs.rs/jacques](https://docs.rs/jacques).

## Minimum Supported Rust Version (MSRV)

Jacques requires Rust 1.88 or later.

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.
