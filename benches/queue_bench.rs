#![allow(missing_docs, clippy::similar_names, clippy::cast_possible_truncation)]

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use std::{
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};
use tokio::{runtime::Runtime, task};

// Import queue implementations
use jacques::{
    QueueError,
    owned::queue,
    pack::queue_pack,
    pointer::pointer_queue,
    traits::{QueueConsumer, QueueProducer},
};

#[cfg(feature = "dev-profiling")]
mod profiling {
    use criterion::profiler::Profiler;
    use pprof::ProfilerGuard;
    use std::{fs::File, path::Path};

    pub struct FlamegraphProfiler<'a> {
        frequency: i32,
        active_profiler: Option<ProfilerGuard<'a>>,
    }

    impl FlamegraphProfiler<'_> {
        #[allow(dead_code)]
        pub const fn new(frequency: i32) -> Self {
            FlamegraphProfiler {
                frequency,
                active_profiler: None,
            }
        }
    }

    impl Profiler for FlamegraphProfiler<'_> {
        fn start_profiling(&mut self, _benchmark_id: &str, _benchmark_dir: &Path) {
            self.active_profiler = Some(ProfilerGuard::new(self.frequency).unwrap());
        }

        fn stop_profiling(&mut self, _benchmark_id: &str, benchmark_dir: &Path) {
            std::fs::create_dir_all(benchmark_dir).unwrap();
            let flamegraph_path = benchmark_dir.join("flamegraph.svg");
            let flamegraph_file = File::create(&flamegraph_path)
                .expect("File system error while creating flamegraph.svg");

            if let Some(profiler) = self.active_profiler.take() {
                profiler
                    .report()
                    .build()
                    .unwrap()
                    .flamegraph(flamegraph_file)
                    .expect("Error writing flamegraph");
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
struct SmallData {
    value: u32,
}

#[derive(Debug, Clone, PartialEq)]
struct LargeData {
    id: u64,
    data: Vec<u8>,
}

impl LargeData {
    fn new(id: u64, size: usize) -> Self {
        Self {
            id,
            data: vec![0u8; size],
        }
    }
}

/// Single-threaded latency benchmark - measures ns per operation
fn bench_latency(c: &mut Criterion) {
    let mut group = c.benchmark_group("latency_ns_per_op");
    group.measurement_time(Duration::from_secs(10));
    group.warm_up_time(Duration::from_secs(3));

    // Test different data types and queue types
    let test_cases = vec![
        ("owned_small", "small"),
        ("pointer_large_1kb", "pointer_1kb"),
        ("pack_small_4queues", "pack_small"),
    ];

    for (name, data_type) in test_cases {
        group.bench_function(name, |b| match data_type {
            "small" => {
                let (producer, consumer) = queue::<SmallData>().capacity(1024).channels().unwrap();
                b.iter(|| {
                    let data = SmallData { value: 42 };
                    producer.push(black_box(data)).unwrap();
                    black_box(consumer.pop().unwrap());
                });
            },
            "pointer_1kb" => {
                let (producer, consumer) = pointer_queue::<LargeData>()
                    .capacity(1024)
                    .channels()
                    .unwrap();
                let data = Arc::new(LargeData::new(1, 1024));
                b.iter(|| {
                    producer.push(black_box(data.clone())).unwrap();
                    black_box(consumer.pop().unwrap());
                });
            },
            "pack_small" => {
                let (producer, consumer) = queue_pack::<SmallData, 4, 16>()
                    .queue_capacity(256)
                    .channels()
                    .unwrap();
                b.iter(|| {
                    let data = SmallData { value: 42 };
                    producer.push(black_box(data)).unwrap();
                    black_box(consumer.pop().unwrap());
                });
            },
            _ => unreachable!(),
        });
    }

    group.finish();
}

/// Multi-threaded throughput benchmark
fn bench_throughput(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let mut group = c.benchmark_group("throughput_ops_per_sec");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(15));
    group.warm_up_time(Duration::from_secs(5));

    // Test configurations: (producers, consumers, ops_per_producer)
    let configs = vec![
        (1, 1, 100_000), // Single producer/consumer baseline
        (2, 2, 50_000),  // Low contention
        (4, 4, 25_000),  // Medium contention
        (8, 8, 12_500),  // High contention
    ];

    for (producers, consumers, ops_per_producer) in configs {
        let total_ops = producers * ops_per_producer;
        let config_name = format!("{producers}p_{consumers}c");

        group.throughput(Throughput::Elements(total_ops as u64));

        // Owned small data
        group.bench_with_input(
            BenchmarkId::new("owned_small", &config_name),
            &(producers, consumers, ops_per_producer),
            |b, &(producers, consumers, ops_per_producer)| {
                b.to_async(&rt).iter(|| async {
                    bench_owned_small(producers, consumers, ops_per_producer).await;
                });
            },
        );

        // Pointer large data
        group.bench_with_input(
            BenchmarkId::new("pointer_large_1kb", &config_name),
            &(producers, consumers, ops_per_producer),
            |b, &(producers, consumers, ops_per_producer)| {
                b.to_async(&rt).iter(|| async {
                    bench_pointer_large(producers, consumers, ops_per_producer).await;
                });
            },
        );

        // Pack small data - 4 queues
        group.bench_with_input(
            BenchmarkId::new("pack_small_4queues", &config_name),
            &(producers, consumers, ops_per_producer),
            |b, &(producers, consumers, ops_per_producer)| {
                b.to_async(&rt).iter(|| async {
                    bench_pack_small(producers, consumers, ops_per_producer).await;
                });
            },
        );

        // Pack small data - 8 queues (for higher contention scenarios)
        if producers >= 4 {
            group.bench_with_input(
                BenchmarkId::new("pack_small_8queues", &config_name),
                &(producers, consumers, ops_per_producer),
                |b, &(producers, consumers, ops_per_producer)| {
                    b.to_async(&rt).iter(|| async {
                        bench_pack_small_8(producers, consumers, ops_per_producer).await;
                    });
                },
            );
        }
    }

    group.finish();
}

async fn bench_owned_small(producers: usize, consumers: usize, ops_per_producer: usize) {
    let (producer, consumer) = queue::<SmallData>().capacity(262_144).channels().unwrap();

    let total_ops = producers * ops_per_producer;
    let consumed = Arc::new(AtomicUsize::new(0));

    // Spawn producers
    let producer_handles: Vec<_> = (0..producers)
        .map(|_| {
            let producer = producer.clone();
            task::spawn(async move {
                for i in 0..ops_per_producer {
                    let data = SmallData { value: i as u32 };

                    loop {
                        match producer.push(data) {
                            Ok(()) => break,
                            Err(QueueError::Full) => task::yield_now().await,
                            Err(e) => panic!("Unexpected queue error: {e:?}"),
                        }
                    }
                }
            })
        })
        .collect();

    // Spawn consumers
    let consumer_handles: Vec<_> = (0..consumers)
        .map(|_| {
            let consumer = consumer.clone();
            let consumed = consumed.clone();
            task::spawn(async move {
                loop {
                    if consumed.load(Ordering::Relaxed) >= total_ops {
                        break;
                    }

                    match consumer.try_pop() {
                        Ok(_) => {
                            consumed.fetch_add(1, Ordering::Relaxed);
                        },
                        Err(QueueError::Empty) => task::yield_now().await,
                        Err(e) => panic!("Unexpected queue error: {e:?}"),
                    }
                }
            })
        })
        .collect();

    // Wait for completion
    for handle in producer_handles {
        handle.await.unwrap();
    }

    while consumed.load(Ordering::Relaxed) < total_ops {
        task::yield_now().await;
    }

    for handle in consumer_handles {
        handle.await.unwrap();
    }
}

async fn bench_pointer_large(producers: usize, consumers: usize, ops_per_producer: usize) {
    let (producer, consumer) = pointer_queue::<LargeData>()
        .capacity(262_144)
        .channels()
        .unwrap();

    let total_ops = producers * ops_per_producer;
    let consumed = Arc::new(AtomicUsize::new(0));

    // Spawn producers
    let producer_handles: Vec<_> = (0..producers)
        .map(|producer_id| {
            let producer = producer.clone();
            task::spawn(async move {
                for i in 0..ops_per_producer {
                    let data = Arc::new(LargeData::new(
                        (producer_id * ops_per_producer + i) as u64,
                        1024,
                    ));

                    loop {
                        match producer.push(data.clone()) {
                            Ok(()) => break,
                            Err(QueueError::Full) => task::yield_now().await,
                            Err(e) => panic!("Unexpected queue error: {e:?}"),
                        }
                    }
                }
            })
        })
        .collect();

    // Spawn consumers
    let consumer_handles: Vec<_> = (0..consumers)
        .map(|_| {
            let consumer = consumer.clone();
            let consumed = consumed.clone();
            task::spawn(async move {
                loop {
                    if consumed.load(Ordering::Relaxed) >= total_ops {
                        break;
                    }

                    match consumer.try_pop() {
                        Ok(_) => {
                            consumed.fetch_add(1, Ordering::Relaxed);
                        },
                        Err(QueueError::Empty) => task::yield_now().await,
                        Err(e) => panic!("Unexpected queue error: {e:?}"),
                    }
                }
            })
        })
        .collect();

    // Wait for completion
    for handle in producer_handles {
        handle.await.unwrap();
    }

    while consumed.load(Ordering::Relaxed) < total_ops {
        task::yield_now().await;
    }

    for handle in consumer_handles {
        handle.await.unwrap();
    }
}

async fn bench_pack_small(producers: usize, consumers: usize, ops_per_producer: usize) {
    let (producer, consumer) = queue_pack::<SmallData, 4, 16>()
        .queue_capacity(65536)
        .channels()
        .unwrap();

    let total_ops = producers * ops_per_producer;
    let consumed = Arc::new(AtomicUsize::new(0));

    // Spawn producers
    let producer_handles: Vec<_> = (0..producers)
        .map(|_| {
            let producer = producer.clone();
            task::spawn(async move {
                for i in 0..ops_per_producer {
                    let data = SmallData { value: i as u32 };

                    loop {
                        match producer.push(data) {
                            Ok(()) => break,
                            Err(QueueError::Full) => task::yield_now().await,
                            Err(e) => panic!("Unexpected queue error: {e:?}"),
                        }
                    }
                }
            })
        })
        .collect();

    // Spawn consumers
    let consumer_handles: Vec<_> = (0..consumers)
        .map(|_| {
            let consumer = consumer.clone();
            let consumed = consumed.clone();
            task::spawn(async move {
                loop {
                    if consumed.load(Ordering::Relaxed) >= total_ops {
                        break;
                    }

                    match consumer.try_pop() {
                        Ok(_) => {
                            consumed.fetch_add(1, Ordering::Relaxed);
                        },
                        Err(QueueError::Empty) => task::yield_now().await,
                        Err(e) => panic!("Unexpected queue error: {e:?}"),
                    }
                }
            })
        })
        .collect();

    // Wait for completion
    for handle in producer_handles {
        handle.await.unwrap();
    }

    while consumed.load(Ordering::Relaxed) < total_ops {
        task::yield_now().await;
    }

    for handle in consumer_handles {
        handle.await.unwrap();
    }
}

async fn bench_pack_small_8(producers: usize, consumers: usize, ops_per_producer: usize) {
    let (producer, consumer) = queue_pack::<SmallData, 8, 16>()
        .queue_capacity(32768)
        .channels()
        .unwrap();

    let total_ops = producers * ops_per_producer;
    let consumed = Arc::new(AtomicUsize::new(0));

    // Spawn producers
    let producer_handles: Vec<_> = (0..producers)
        .map(|_| {
            let producer = producer.clone();
            task::spawn(async move {
                for i in 0..ops_per_producer {
                    let data = SmallData { value: i as u32 };

                    loop {
                        match producer.push(data) {
                            Ok(()) => break,
                            Err(QueueError::Full) => task::yield_now().await,
                            Err(e) => panic!("Unexpected queue error: {e:?}"),
                        }
                    }
                }
            })
        })
        .collect();

    // Spawn consumers
    let consumer_handles: Vec<_> = (0..consumers)
        .map(|_| {
            let consumer = consumer.clone();
            let consumed = consumed.clone();
            task::spawn(async move {
                loop {
                    if consumed.load(Ordering::Relaxed) >= total_ops {
                        break;
                    }

                    match consumer.try_pop() {
                        Ok(_) => {
                            consumed.fetch_add(1, Ordering::Relaxed);
                        },
                        Err(QueueError::Empty) => task::yield_now().await,
                        Err(e) => panic!("Unexpected queue error: {e:?}"),
                    }
                }
            })
        })
        .collect();

    // Wait for completion
    for handle in producer_handles {
        handle.await.unwrap();
    }

    while consumed.load(Ordering::Relaxed) < total_ops {
        task::yield_now().await;
    }

    for handle in consumer_handles {
        handle.await.unwrap();
    }
}

#[cfg(feature = "dev-profiling")]
criterion_group! {
    name = benches;
    config = Criterion::default()
        .significance_level(0.01)
        .noise_threshold(0.05)
        .with_profiler(profiling::FlamegraphProfiler::new(100));
    targets = bench_latency, bench_throughput
}

#[cfg(not(feature = "dev-profiling"))]
criterion_group!(benches, bench_latency, bench_throughput);

criterion_main!(benches);
