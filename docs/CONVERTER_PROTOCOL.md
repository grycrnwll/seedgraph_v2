# Converter protocol

PDF conversion runs in a separate, locked Python environment (`converter/`, see
[DEPENDENCY_POLICY.md](DEPENDENCY_POLICY.md)). The app talks to it through one narrow
contract: **PDF path in, markdown + page/metadata info out.** The converter never
imports seedgraph and never touches a database; the app saves what it returns.

## Processes

- One **warm converter** per GPU. Each is a child process started by the app:

  ```
  uv run --frozen --project converter python -m sg_converter.worker
  ```

  `--frozen` matters: plain `uv run` re-locks and re-syncs the environment.
- The app pins each child to one GPU with `CUDA_VISIBLE_DEVICES=<index>`. GPUs are
  counted with `nvidia-smi -L` (the app never imports torch). `SEEDGRAPH_MARKER_SLOTS`
  overrides the number of converters; with no GPU there is one unpinned converter.
- A child loads the Marker models once, then serves many PDFs, one at a time.
- It starts when there is work and exits after `SEEDGRAPH_MARKER_IDLE_SECONDS`
  (default 600) without work.
- No network port: the pipe is the child's stdin/stdout.

## Wire format

- Newline-delimited JSON, UTF-8, one object per line, in both directions.
- stdin carries app → worker messages; the worker's **original** stdout carries
  worker → app messages.
- At start-up the worker duplicates its stdout handle for the protocol, then points
  file descriptor 1 and `sys.stdout` at stderr. Library output (tqdm bars, prints,
  C-level writes) therefore lands on stderr and cannot corrupt the protocol.
- stderr is free-form log text. The app drains it continuously and keeps the tail for
  error messages.

## Messages (protocol version 1)

### worker → app: `ready` (once, after the models are loaded)

```json
{"type": "ready", "protocol": 1,
 "converter": {"name": "marker-pdf", "version": "1.10.2"},
 "python": "3.13.x", "torch": "2.11.0+cu128", "cuda_device": "NVIDIA GeForce RTX 5080",
 "load_seconds": 14.2}
```

The app refuses a worker whose `protocol` differs from its own. It also refuses one whose
`converter.version` differs from the marker-pdf version pinned in `converter/uv.lock`.
That pinned version is what each paper's `conversion_runs.converter_version` records.

### app → worker: `convert`

```json
{"type": "convert", "id": "r1", "pdf_path": "C:\\abs\\path\\paper.pdf",
 "options": {"paginate_output": true, "force_ocr": false, "output_format": "markdown"}}
```

Missing options take the defaults shown. `use_llm` is not part of the protocol. The
app refuses LLM-assisted conversion before sending anything.

### worker → app: `result`

Success:

```json
{"type": "result", "id": "r1", "ok": true,
 "markdown": "...", "page_count": 12, "metadata": {"...": "Marker metadata"},
 "warnings": [], "timings": {"convert_seconds": 21.7}}
```

- `page_count` comes from the PDF itself (pypdfium2), not from the markdown. With
  `paginate_output` the markdown carries one `{N}------…` delimiter line per page
  (0-based).
- `metadata` is Marker's metadata dict. The app stores it verbatim as `meta.json`.

Failure, for this PDF only (the worker stays up):

```json
{"type": "result", "id": "r1", "ok": false, "error": "RuntimeError: <message>"}
```

### app → worker: `ping` / worker → app: `pong`

```json
{"type": "ping"}
{"type": "pong"}
```

### app → worker: `shutdown`

```json
{"type": "shutdown"}
```

The worker exits with code 0. It also exits cleanly when stdin reaches EOF (the app
went away).

### worker → app: `error`

The worker sends this for a message it could not parse or does not know:

```json
{"type": "error", "id": "b1", "error": "unknown message type: 'bogus'"}
```

## Failure handling (app side)

- **Per-PDF timeout:** `SEEDGRAPH_MARKER_TIMEOUT` seconds (default 1800). If a
  conversion does not finish in time, the app kills the child's whole process tree
  (`taskkill /T` on Windows), and that PDF fails.
- **Child dies mid-PDF:** that PDF fails. The error includes the child's exit code and
  the tail of its stderr.
- **Never becomes ready:** if the child takes longer than
  `SEEDGRAPH_MARKER_READY_TIMEOUT` (default 900 s) to load, it is killed and the job
  fails.
- After any of these, the next job starts a fresh child, and the queue continues.
- Crashes and timeouts are not retried with OCR. An error *result* still triggers the
  usual OCR fallback.

## Versioning

`protocol` is an integer that is bumped on any incompatible change to the messages
above. Adding optional fields does not bump it. Readers ignore fields they don't know.
