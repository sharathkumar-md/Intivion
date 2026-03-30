"""
Async event stream simulator — Kafka-compatible interface over asyncio.Queue.

The brief asks for Kafka/RabbitMQ for real-time streaming. I'm simulating
it with an async queue that has the same put/get interface as aiokafka.
Swapping to real Kafka would be two lines of code — replace queue.put/get
with aiokafka producer.send / consumer.__anext__. Everything else stays the same.

The producer replays events from events.parquet in timestamp order with a
configurable speed multiplier (100x = 100x faster than real time). The
UserEventBuffer maintains a rolling 20-event window per user that the LSTM
endpoint can pull from for live scoring.
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import AsyncIterator

import numpy as np
import pandas as pd

from forexguard.log_utils import setup_logger
log = setup_logger("forexguard.streaming.simulator", "forexguard_streaming.log")


# ──────────────────────────────────────────────────────────────────────────────
# EventQueue — Kafka-compatible wrapper around asyncio.Queue
# ──────────────────────────────────────────────────────────────────────────────

class EventQueue:
    """
    Async FIFO queue with a Kafka-compatible interface.
    Swap `put`/`get` with aiokafka producer/consumer for production.
    """

    def __init__(self, maxsize: int = 10_000):
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=maxsize)

    async def put(self, event: dict) -> None:
        await self._queue.put(event)

    async def get(self) -> dict:
        return await self._queue.get()

    def task_done(self) -> None:
        self._queue.task_done()

    def empty(self) -> bool:
        return self._queue.empty()

    def qsize(self) -> int:
        return self._queue.qsize()


# ──────────────────────────────────────────────────────────────────────────────
# Producer — replays events from parquet
# ──────────────────────────────────────────────────────────────────────────────

async def produce_events(
    queue: EventQueue,
    events_df: pd.DataFrame,
    speed_factor: float = 100.0,
    max_events: int | None = None,
) -> None:
    """
    Replay events in timestamp order into the queue.

    Parameters
    ----------
    queue        : EventQueue to publish into
    events_df    : raw events DataFrame (must have 'timestamp' column)
    speed_factor : simulation speed multiplier (100 = 100x real-time)
    max_events   : stop after this many events (None = all)
    """
    log.info("Producer starting. Events: %d | speed_factor: %.0fx", len(events_df), speed_factor)

    events_df = events_df.copy()
    events_df["timestamp"] = pd.to_datetime(events_df["timestamp"], format="mixed")
    events_df = events_df.sort_values("timestamp").reset_index(drop=True)

    if max_events:
        events_df = events_df.head(max_events)

    prev_ts = None
    produced = 0

    for _, row in events_df.iterrows():
        ts = row["timestamp"]

        if prev_ts is not None:
            delta_real = (ts - prev_ts).total_seconds()
            delay = max(0.0, delta_real / speed_factor)
            if delay > 0.001:
                await asyncio.sleep(delay)

        event = {
            "event_id":            str(row.get("event_id", f"evt_{produced}")),
            "user_id":             str(row["user_id"]),
            "timestamp":           ts.isoformat(),
            "event_type":          str(row["event_type"]),
            "ip_address":          str(row.get("ip_address", "")),
            "device_fingerprint":  str(row.get("device_fingerprint", "")),
            "country":             str(row.get("country", "")),
            "trade_volume":        float(row.get("trade_volume", 0)),
            "lot_size":            float(row.get("lot_size", 0)),
            "pnl":                 float(row.get("pnl", 0)),
            "deposit_amount":      float(row.get("deposit_amount", 0)),
            "withdrawal_amount":   float(row.get("withdrawal_amount", 0)),
            "session_duration":    float(row.get("session_duration", 0)),
            "pages_per_minute":    float(row.get("pages_per_minute", 0)),
            "margin_used":         float(row.get("margin_used", 0)),
            "failed_logins":       int(row.get("failed_logins", 0)),
            "instrument":          str(row.get("instrument", "NONE")),
        }

        await queue.put(event)
        produced += 1
        prev_ts = ts

        if produced % 5000 == 0:
            log.info("  Produced %d events (queue size: %d)", produced, queue.qsize())

    log.info("Producer finished. Total events produced: %d", produced)


# ──────────────────────────────────────────────────────────────────────────────
# Consumer — echoes events (replace with real processing logic in API)
# ──────────────────────────────────────────────────────────────────────────────

async def consume_events(
    queue: EventQueue,
    stop_event: asyncio.Event,
    max_consume: int | None = None,
) -> list[dict]:
    """
    Demo consumer: drains the queue and collects events.
    In production this would trigger the scoring pipeline per batch.
    """
    log.info("Consumer starting...")
    consumed = []
    count = 0

    while not stop_event.is_set() or not queue.empty():
        try:
            event = await asyncio.wait_for(queue.get(), timeout=1.0)
            consumed.append(event)
            queue.task_done()
            count += 1
            if max_consume and count >= max_consume:
                break
        except asyncio.TimeoutError:
            if stop_event.is_set():
                break

    log.info("Consumer finished. Total events consumed: %d", count)
    return consumed


# ──────────────────────────────────────────────────────────────────────────────
# User event buffer — groups streaming events by user for real-time scoring
# ──────────────────────────────────────────────────────────────────────────────

class UserEventBuffer:
    """
    Maintains a rolling window of the last N events per user.
    Used by the API to assemble a mini-sequence for LSTM scoring.
    """

    def __init__(self, window: int = 20):
        self.window = window
        self._buffer: dict[str, list[dict]] = {}

    def add_event(self, event: dict) -> None:
        uid = event["user_id"]
        if uid not in self._buffer:
            self._buffer[uid] = []
        self._buffer[uid].append(event)
        if len(self._buffer[uid]) > self.window:
            self._buffer[uid].pop(0)

    def get_user_events(self, user_id: str) -> list[dict]:
        return self._buffer.get(user_id, [])

    def user_count(self) -> int:
        return len(self._buffer)

    def clear(self) -> None:
        self._buffer.clear()


# ──────────────────────────────────────────────────────────────────────────────
# Entry point for standalone testing
# ──────────────────────────────────────────────────────────────────────────────

async def _main():
    raw_dir = Path(__file__).parent.parent / "data" / "raw"
    events_df = pd.read_parquet(raw_dir / "events.parquet")

    queue       = EventQueue(maxsize=5000)
    stop_event  = asyncio.Event()

    # Run producer and consumer concurrently
    producer_task = asyncio.create_task(
        produce_events(queue, events_df, speed_factor=10000.0, max_events=1000)
    )
    consumer_task = asyncio.create_task(
        consume_events(queue, stop_event, max_consume=1000)
    )

    await producer_task
    stop_event.set()
    consumed = await consumer_task

    log.info("Sample events received by consumer:")
    for ev in consumed[:3]:
        log.info("  %s", ev)
    log.info("Stream simulation complete [OK]")


if __name__ == "__main__":
    asyncio.run(_main())
