# Inference Engineering Plan

This note captures a proposed plan for improving inference in PricerProject. It is a planning document; no implementation changes are implied.

## Goal

Make pricing inference faster, more reliable, and easier to operate while checking that performance changes preserve pricing quality.

## Plan

1. **Establish a baseline.** Run a fixed set of representative listings through the current flow. Record end-to-end and per-stage latency, throughput at several concurrency levels, memory use, failures, and outputs. Report cold starts separately from warmed-up runs.
2. **Build an independent quality reference.** Prepare labeled products that do not overlap training products. Track pricing error alongside performance; until then, report speed and reliability only, not pricing accuracy.
3. **Locate the bottleneck.** Attribute time to listing preparation, embeddings, Chroma/BM25 retrieval, LLM inference, neural prediction, and ensemble logic. Prioritize work using observed workload impact.
4. **Optimize one area at a time.** Consider model and encoder reuse, batching neural and embedding requests, caching repeated embeddings or retrieval results, trimming unnecessary LLM context, and tuning concurrency and timeouts. Compare each change with the baseline.
5. **Define failure behavior.** Exercise slow, unavailable, and malformed LLM and retrieval responses. Set timeouts and fallbacks, and make partial results and uncertainty visible.
6. **Compare serving options.** Benchmark Ollama, vLLM, and/or the LiteLLM gateway on the same requests. Compare cold and warm latency, throughput, memory, and failures on the target hardware.
7. **Set acceptance targets and report results.** Agree on latency, resource, reliability, and allowable quality-change targets. Re-run the benchmark and document the hardware and configuration for reproducibility.

## First milestone

Produce a baseline report and a small independent labeled set. Use these to choose the first optimization and verify that it does not degrade pricing quality beyond the agreed limit.

## Current evidence and limitations

The repository has a 10-request warm, sequential LiteLLM smoke measurement, which checks basic latency and nonempty responses. It does not establish concurrency capacity or pricing accuracy. The README currently notes that there is no independently labeled pricing holdout result checked in.
