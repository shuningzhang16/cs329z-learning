# CS329Z Learning

个人学习仓库：逐步完成 CS329Z Assignment 1，记录实现思路、测试结果和遇到的问题。
课程起始代码来自 [cs329z/assignment1-harness](https://github.com/cs329z/assignment1-harness)，
保留原始 Git 历史和 [MIT License](LICENSE)。

目前完成了本地环境、Codex CLI 模型接入和适配测试，课程题目的实现仍在进行中。
每次完成一个小目标后提交对应代码和笔记，详细记录见 [学习日志](LEARNING_LOG.md)。

## 学习进度

- [x] 准备 Python 环境与邮件数据集
- [x] 将模型接口接入 Codex CLI，保留缓存、账本与 ScriptedLM
- [x] 验证适配层：9 项测试通过
- [ ] 阅读 priority 题目和测试，完成邮件优先级分类
- [ ] 完成 daily digest、BM25 与邮件问答
- [ ] 逐步实现工具协议、工具注册与 Agent 循环
- [ ] 完成记忆、压缩、guardrails 与综合评估

## 每天提交

先完成一个小目标，再把当天的改动、测试结果和下一步补充到 `LEARNING_LOG.md`。
在 VS Code 的“源代码管理”中检查差异，暂存本次修改，填写提交说明并推送即可。
命令行也可以按下面的步骤操作（把示例文件名和说明换成本次实际内容）：

```powershell
git status
git diff
git add cs329z_hw1/adapters.py LEARNING_LOG.md
git commit -m "Implement priority classification"
git push
```

实现放在新文件中时，也要把相应文件加入暂存区。测试和阅读笔记同样可以独立提交。

## Assignment 1: Building an Agentic Harness

The following sections describe the course setup and commands, adapted for this
local learning repository.

**Start by reading the handout, [`cs329z_assignment1_harness.pdf`](cs329z_assignment1_harness.pdf), in this directory.**
It states every problem, what is graded, and the design decisions that
are yours to make. This file covers setup, the commands you will run, and
where things are.


## Setup

1. Install `uv`, which manages Python and the dependencies:
   https://docs.astral.sh/uv/getting-started/installation/

   The assignment needs Python 3.11 or newer. `uv` installs one if you do
   not have it. macOS, Linux and Windows are supported. On Windows, also
   install Git for Windows (https://git-scm.com/download/win, default
   options): the terminal sandbox runs commands with the `bash.exe` it
   ships, and there is no operating-system jail on Windows (see "The
   terminal sandbox" below). WSL works too and gives you the Linux jail.

2. Sign in to Codex CLI with your ChatGPT account:

   ```sh
   codex --version
   codex login
   codex login status
   ```

   This local learning version uses `codex exec`, not an API key. Codex CLI
   must be on `PATH`; this integration was tested with version 0.144.6.
   If you are already logged in with ChatGPT, skip `codex login`.
   `.env` is optional and gitignored; `.env.example` lists model, timeout,
   and evaluation settings. Test Python -> Codex with:

   ```sh
   uv run python -c "from cs329z_hw1.llm import LM; print(LM()([{'role':'user','content':'Only reply HELLO'}]))"
   ```


3. Unpack the email archive. It ships in the repository as
   `data/emails.jsonl.gz` (19 MB). This command verifies its checksum and
   unpacks it to `data/emails/emails.jsonl` (58 MB):

   ```sh
   uv run python data/download.py
   ```

4. Run the tests. Course tests initially fail with `NotImplementedError`
   until you implement the adapters; provider integration tests should pass:

   ```sh
   uv run pytest
   ```

Run every command from the repository root with `uv run`. The first
`uv run` creates the environment in `.venv/`.

## Tests

There are two kinds of tests, in the same files under `tests/`.

**Deterministic tests** run by default and make no model calls. Where your
code needs a model, they pass a `ScriptedLM`, a stand-in that replays a fixed
list of replies.

```sh
uv run pytest                          # all deterministic tests
uv run pytest tests/test_priority.py   # one problem
```

**Live tests** call Codex using your ChatGPT plan and compare results against
the original thresholds in `tests/thresholds.py`. They consume plan usage.
The changed model/backend means these are learning checks, not an equivalent
reproduction of the course's original model-based grading.

```sh
uv run pytest -m live -s                          # all live tests
uv run pytest tests/test_priority.py -m live -s   # one problem
```

A live test can take several minutes (`priority` makes 60 model calls),
and pytest prints nothing while it waits. Starting a CLI per uncached call
adds latency; the original test can run multiple calls concurrently.
`-s` makes each test print its score and the items your code got wrong,
whether or not it passes.

While you iterate, `CS329Z_EVAL_SLICE=N` runs each live test on its first
N items only. You can set it on the command line or in `.env`:

```sh
CS329Z_EVAL_SLICE=5 uv run pytest tests/test_priority.py -m live -s
```

## Seeing what your pipelines produce

The tests report scores. To read the output itself, run a pipeline on real
data:

```sh
uv run python -m cs329z_hw1.show priority em-12167              # label and reason for one email
uv run python -m cs329z_hw1.show digest 2001-06-22 --labels --limit 30   # the digest, plus each label
uv run python -m cs329z_hw1.show bm25 "larchfield audit"                 # top emails for a query, with scores
uv run python -m cs329z_hw1.show email_qa "Who leads the Basin Analytics move?"
uv run python -m cs329z_hw1.show search_docs "parental leave"
uv run python -m cs329z_hw1.show memory .lm_cache/chat_memory             # what chat.py's agent has stored
uv run python -m cs329z_hw1.show memory                                   # each persona's memory after an evaluation run
```

Each prints the result and the time of model calls. Codex does not report
USD charges; displayed dollar zeros are placeholders. Cached repeats use
no additional model quota.

## Model calls and spending

All model calls go through `cs329z_hw1.llm.LM`: a list of
`{"role", "content"}` messages in, a string out.

```python
from cs329z_hw1.llm import LM

lm = LM()            # Codex model, gpt-5.6-sol
judge = LM("judge")  # same verified model by default; override in .env
text = lm([{"role": "user", "content": "Say hello."}])
```

`_complete()` serializes the ordered system/user/assistant messages as JSON
in one prompt. The CLI returns the next assistant reply through
`--output-last-message`; stdout JSON events provide token usage. This is a
chat-role emulation, not native Chat Completions role handling.

Calls run in a temporary directory with `--sandbox read-only`, personal config
ignored, and native command execution, apps, plugins and agent delegation
disabled. Codex interprets the conversation as text; your course harness still
implements and executes its own tool protocol. The original `ScriptedLM` and
course adapters are unchanged. The original provider file is backed up as
`cs329z_hw1/llm_openrouter_backup.py`.

The wrapper retains:

- **Cache.** Each reply is stored under `.lm_cache/`, keyed on the provider, model,
  the sampling settings and the exact messages. Repeating an identical call
  returns the stored reply without calling Codex. Old provider cache entries
  remain on disk but are not reused by Codex.
- **Ledger.** Each call that reaches the provider is appended to
  `.lm_cache/ledger.jsonl` with its real CLI token counts, time and provider.
  `cost_usd=0` and `cost_source=not_reported_chatgpt` mean the USD cost is
  unavailable, not that the call consumes no quota. Print the totals
  by model and by tag (the test suite tags each call with the test name)
  with:

  ```sh
  uv run python -m cs329z_hw1.llm
  ```

- **Budget.** The original `CS329Z_BUDGET_USD` ledger check remains and still
  raises `BudgetExceeded` when recorded dollar spending reaches the cap.
  It cannot limit ChatGPT quota, because Codex does not report dollar charges.
  Account limits still apply. Use a small `CS329Z_EVAL_SLICE` while learning.

`temperature` and `max_tokens` remain accepted and are part of cache keys,
but the CLI has no equivalent per-call controls, so they are not enforced.
`OutputTruncated` remains available for course compatibility; CLI failures
and timeouts raise `RuntimeError` and do not create cached replies.
`CS329Z_REASONING_EFFORT` controls Codex reasoning effort (default `low`), and
`CS329Z_CODEX_TIMEOUT_SECONDS` controls the subprocess timeout (default 180).

Official references: [non-interactive mode](https://learn.chatgpt.com/docs/non-interactive-mode)
and [CLI configuration](https://learn.chatgpt.com/docs/config-file/config-reference).

Deleting `.lm_cache/` deletes both the cache and the ledger.

## How to work a problem

Your code goes anywhere inside the `cs329z_hw1` package, and the tests
reach it only through the functions in `cs329z_hw1/adapters.py`, which ship
as stubs that raise `NotImplementedError`.

For each problem:

1. Read the problem in the handout.
2. Open the adapter it names in `cs329z_hw1/adapters.py`. The adapter's type
   hints and docstring are the exact signature the tests use.
3. Implement the functionality in your package, for example in
   `cs329z_hw1/pipelines/priority.py`.
4. Replace the stub's body with a call into your code:

   ```python
   def run_priority(email, lm):
       from cs329z_hw1.pipelines.priority import classify_priority
       return classify_priority(email, lm)
   ```

5. Run that problem's tests, deterministic first and then live.

The deterministic tests are the complete list of the error handling we
require (an empty reply, a reply in the wrong format, a cut-off tool call,
a tool that raises). Nothing is graded that is not in a test or asked for
in the design memo.

The types that cross the adapter boundary (`Email`, `ToolSpec`, `ToolCall`,
`ToolResult`, `ParsedResponse`, `AgentConfig`, `AgentResult`, and others) are
defined in `cs329z_hw1/types.py`. Use them as they are.

## Repository layout

Files marked *provided* are course code, which grading replaces with its
own copies, so read them but do not edit them. Files marked *read* define
interfaces you build on or rules the tests check, and each is worth
reading before the problem that first uses it.

```
README.md  pyproject.toml  .env.example
cs329z_hw1/                               your package: every file you write goes here
  adapters.py           yours     read  the only file the tests call; starts as stubs
  pipelines/            yours           Part 1 (starts empty)
  agent/                yours           Part 2 (starts empty)
  llm.py                provided  read  LM, ScriptedLM, cache, ledger, budget
  types.py              provided  read  data types shared by your code and the tests
  tokens.py             provided  read  count_tokens, count_message_tokens (4 characters per token)
  tokenizer.py          provided        tokenize(), the tokenizer BM25 must use
  data.py               provided  read  load_emails(), emails_on(), email_text(), load_docs()
  sandbox.py            provided  read  run_terminal(), make_workspace()
  cardinal.py           provided  read  names and argument schemas of the agent tools
  user.py               provided  read  UserIO, ScriptedUser, ConsoleUser
  chat.py               provided        terminal chat with your agent
  show.py               provided        run one Part 1 pipeline on real data and print the result
  simulation/           provided        simulated users, judge, evaluation runner (see its README.md)
tests/                                    provided: read, do not edit; discarded at grading
  conftest.py           provided
  helpers.py            provided
  thresholds.py         provided        pass thresholds for the live tests
  test_*.py             provided  read  one file per problem; read the one you are on
  fixtures/             provided        gold labels, scripted sessions, personas
data/
  docs/                 provided        Cardinal Energy documents
  emails.jsonl.gz       provided        the email archive, compressed
  download.py           provided        verifies and unpacks the archive
  source.json           provided        checksum of the archive
  priority_rubric.md    provided  read  how the gold priority labels were assigned
  emails/                         unpacked by download.py, gitignored
```


You may add files and subpackages inside `cs329z_hw1/`. Grading copies
your whole `cs329z_hw1/` package into a clean starter and discards
everything else, so a helper you put under `tests/` or anywhere outside the
package is not there when we grade. Do not add dependencies to
`pyproject.toml`; NumPy is already one, and you may use it anywhere.

## The terminal sandbox

`cs329z_hw1.sandbox.run_terminal(cmd)` runs one shell command for your
agent. Read `cs329z_hw1/sandbox.py` before you use it. Its module docstring
lists what each layer stops and what it does not. One of those layers is an
operating-system jail that is not available on every machine. To see which
one your machine uses:

```sh
uv run python -c "from cs329z_hw1 import sandbox; print(sandbox.jail())"
```

This prints `sandbox-exec` (macOS), `bwrap` (Linux with bubblewrap
installed and permitted), or `none`. With `none`, a command that runs
`python3` can reach the network and any file your account can. On Linux you
can install bubblewrap with your package manager (for example
`sudo apt install bubblewrap`).

On Windows it always prints `none`: there is no OS jail there, so the
allowlist, the scrubbed environment and the timeout are the only limits.
The shell and the allowed programs (`grep`, `sort`, `awk` and the rest)
come from Git for Windows, so it must be installed; without it every
command is rejected with a message that says so. Commands are written as on
Linux (`grep`, `wc -l`, pipes, `python3 -c '...'`); `python3` is the Python
that `uv` installed, without the project's packages. Line endings in a
command's output are normalized to `\n`.

Each command runs in a workspace directory that holds `emails.jsonl` and
`docs/`. Here-documents (`<<EOF`) are rejected; multi-line Python goes in
`python3 -c '...'`.

## The simulated-user evaluation

```sh
uv run python -m cs329z_hw1.simulation --slice 3   # the first 3 personas
uv run python -m cs329z_hw1.simulation             # all public personas
```

This plays each persona in `tests/fixtures/personas/` against the agent
behind `run_agent_session` and saves the conversations under `runs/`. Pass a
directory as the first argument to use other personas. See
`uv run python -m cs329z_hw1.simulation --help` for the options.

## Chatting with your agent

Once `run_agent_session` works, you can talk to your agent in the terminal.

```sh
uv run python -m cs329z_hw1.chat [--mode confirm|auto] [--memory-dir DIR]
```

## Acknowledgements

This assignment was designed and built by Michael Ryan for Stanford's
CS329Z Engineering AI Agents course with lots of help from Claude Fable 5.1.
We owe a debt of gratitude to the various people who shared thoughts and
feedback on this assignment. In particular we would like to highlight Houjun
(Jack) Liu, Owen Queen, Shreyas Sharma, and Anantharaman Iyer for completing a
preliminary version of the assignment and offering invaluable feedback.

If you would like to use this assignment for your course you are welcome to
do so with brief attribution. It is released under the MIT license (see
`LICENSE`).
