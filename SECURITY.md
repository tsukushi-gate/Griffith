# Security Policy

## Reporting a Vulnerability

Please **do not** open a public GitHub issue for security problems.

Report vulnerabilities privately to **masato.akiho@magicdelta.ai**.

Please include, where possible:

- A description of the issue and its impact.
- The affected version (`griffith --version`) and environment.
- Minimal steps to reproduce, or a proof-of-concept.
- Any suggested mitigation.

### What to expect

| Stage | Target |
| --- | --- |
| First response (acknowledgement) | within 72 hours |
| Initial assessment / severity triage | within 7 days |
| Fix or mitigation plan communicated | within 30 days |

We will keep you informed of progress and will credit you in the release notes
unless you prefer to remain anonymous. Please give us a reasonable opportunity
to release a fix before public disclosure.

## Supported Versions

| Version | Supported |
| --- | --- |
| 0.1.x | ✅ |
| < 0.1 | ❌ |

0.1.x is the first public release line (0.1.0 was released on).
Griffith is pre-1.0 software. Security fixes are applied to the latest release
only; there are currently no long-term support branches.

## Security Notes for Users

These are not vulnerabilities in Griffith, but properties of the formats and
tools it works with. Please read them before deploying Griffith.

### Do not load untrusted checkpoints or IR files

- **`.pt` / `.pth` checkpoints use Python `pickle`.** Loading a checkpoint
  executes arbitrary code contained in it. Only load checkpoints you produced
  yourself or obtained from a source you trust. Prefer `weights_only=True` when
  calling `torch.load` where the checkpoint format permits it.
- **IR JSON files describe a circuit that Griffith will build, simulate, and
  export.** A hostile IR file can specify pathological shapes that exhaust
  memory or CPU, and can drive code generation (Verilog/VHDL) whose output you
  might then synthesize. Validate untrusted IR (`griffith_logic.ir.validate`)
  and treat generated HDL from an untrusted source as untrusted input.
- The same applies to datasets, teacher models, and any other artifact fetched
  from a third party.

### The editor and visualizer backends are for trusted local use only

The visual editor and the visualizer backend (development-only tools that ship
with the development repository, not with the published package) are built for a
**single trusted user on a local machine**. They are **not hardened for exposure
to a network**:

- They have **no authentication or authorization** of any kind.
- They read and write files on the host, and they can **execute Python from the
  workspace** (the "import a model definition" feature imports and runs a Python
  module you point at, and training runs execute in-process).
- They are meant to be bound to the loopback interface and driven by the person
  who started them.

**Do not bind them to a public network** (`--host 0.0.0.0` and equivalents), do
not expose them through a reverse proxy or tunnel, and do not run them as a
shared multi-user service.

Both backends apply defence-in-depth measures so that "it is only local" actually
holds, rather than being an assumption:

| Measure | Behaviour |
| --- | --- |
| Default bind address | Loopback (`127.0.0.1`) only. Binding to all interfaces requires an explicit flag and prints a warning. |
| `Host` header check | Requests whose `Host` is not a loopback name are rejected with **421**. This blocks DNS rebinding, where a page on the internet resolves its own hostname to `127.0.0.1` and then talks to your local server. |
| `Origin` header check | State-changing requests (`POST`/`PUT`/`DELETE`/`PATCH`) and WebSocket handshakes from a non-loopback `Origin` are rejected with **403** (CSRF and cross-site WebSocket hijacking). Requests with no `Origin` — non-browser clients such as `curl` — are allowed. |
| CORS | Restricted to the local frontend dev-server origins, with an explicit method and header allowlist. |
| Path containment | Every filesystem path accepted by the API is resolved and must stay inside the repository working directory. `../` traversal, absolute paths outside it, symlinks pointing out of it, and NUL bytes are rejected with **400**. |
| Checkpoint loading | `torch.load` runs with `weights_only=True`. Checkpoints that pickle a whole model object are **refused by default**; see below. |
| Upload limits | Dataset uploads are read in bounded chunks and rejected with **413** above the limit; ZIP archives are also checked against their *uncompressed* total size (ZIP bombs). |

Two environment variables control the exceptions:

- **`GRIFFITH_EDITOR_ALLOWED_HOSTS`** — comma-separated extra host names accepted
  in the `Host` and `Origin` headers. Needed only if you deliberately bind to a
  non-loopback interface (for example when the backend runs inside a container or
  a remote development box that you reach by name). Wildcard bind addresses such
  as `0.0.0.0` are ignored, because they are not host names. Adding a host here
  widens the DNS-rebinding and CSRF surface, so add the narrowest name that works.
- **`GRIFFITH_ALLOW_UNSAFE_CKPT=1`** — opts in to `torch.load` with
  `weights_only=False`, i.e. to executing the pickle inside a checkpoint. Without
  it, a checkpoint that stores a whole pickled model is refused with an
  explanatory error. Prefer the safe checkpoint layout,
  `{'state_dict': ..., 'config': ...}`, which is rebuilt by code from the
  `config` description and never unpickles arbitrary objects. Set this variable
  only for checkpoints you produced yourself.

- **`GRIFFITH_EDITOR_MAX_UPLOAD_MB`** — dataset upload limit in MiB (default 512).

### Code generation and hardware export

Generated Verilog/VHDL is produced from the IR without sandboxing. Review
generated HDL before feeding it into a synthesis toolchain, particularly if the
IR did not originate from you.
