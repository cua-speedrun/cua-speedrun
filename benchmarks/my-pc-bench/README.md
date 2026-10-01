# MyPCBench

This benchmark contains 38 representative tasks from MyPCBench. The source
pins the task definitions, persona desktop image, and canonical Gemini
trajectory evaluator. Evaluations use 100 action batches and uniformly
average rubric scores.

Run with a Gemini API key for the evaluator:

```bash
export MYPCBENCH_JUDGE_API_KEY="your-api-key"
cua-speedrun benchmark --dataset my-pc-bench --agent qwen3vl
```

The prebuilt desktop image is imported from Docker Hub and cached in Modal.
The judge key stays in the
evaluator process and is not exposed to the agent or desktop.
