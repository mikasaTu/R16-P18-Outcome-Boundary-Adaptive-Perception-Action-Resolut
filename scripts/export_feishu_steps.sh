#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
output_root="${repo_root}/docs/feishu"
mkdir -p "${output_root}"

export_doc() {
  local step="$1"
  local kind="$2"
  local wiki_token="$3"
  local object_token="$4"
  local output_name="$5"
  local source_url="https://icnbwz7kd1ui.feishu.cn/wiki/${wiki_token}"
  local temporary_json
  local revision

  temporary_json="$(mktemp)"
  trap 'rm -f "${temporary_json}"' RETURN

  lark-cli docs +fetch \
    --api-version v2 \
    --as user \
    --doc "${source_url}" \
    --doc-format markdown \
    --detail simple \
    --format json >"${temporary_json}"

  if [[ "$(jq -r '.ok' "${temporary_json}")" != "true" ]]; then
    jq . "${temporary_json}" >&2
    return 1
  fi

  revision="$(jq -r '.data.document.revision_id' "${temporary_json}")"
  mkdir -p "${output_root}/${step}"

  {
    printf '%s\n' '---'
    printf 'feishu_title: "%s"\n' "${kind}"
    printf 'feishu_url: "%s"\n' "${source_url}"
    printf 'feishu_wiki_token: "%s"\n' "${wiki_token}"
    printf 'feishu_object_token: "%s"\n' "${object_token}"
    printf 'feishu_revision: %s\n' "${revision}"
    printf '%s\n\n' '---'
    # Feishu's Markdown export may emit hard-break spaces. Normalize them so
    # repository whitespace checks remain deterministic across CLI versions.
    jq -r '.data.document.content' "${temporary_json}" | sed -E 's/[[:blank:]]+$//'
  } >"${output_root}/${step}/${output_name}"

  printf '%s\t%s\t%s\t%s\n' \
    "${step}" "${kind}" "${revision}" "${output_root}/${step}/${output_name}"
}

export_doc step1 step1 SbJ3wiILHig6uzk6KXWc6wNqnWn S4LPdnINBoBKvSxLV7Bc7xRPno0 step.md
export_doc step1 实验报告 MCkMwHrYpiQZa2kh3H2c7SWhnnd B2ledyTWSo1XjzxCn0bcZAjGntg experiment-report.md
export_doc step2 step2 GbKbwE5eai5l72kje1NcJpn8n5f ASajdohnUoB7s9xwvBqcqkyRnEc step.md
export_doc step2 实验报告 WytawLHgUiNZr6k61UqcpZmQnZc J5z8di0Udo4cOyxntsjcIzopnzh experiment-report.md
export_doc step3 step3 JEhpwKVxIinKlHkxaCFceTC0n1Z RNkWdXQGuoiICHxSPjtcwqqxn2c step.md
export_doc step3 实验报告 SXEuwFuh5iRjdNkey4IckpnWngd MuvJdSQQrophgYx3qgCciJIwnrc experiment-report.md
export_doc step4 step4 Vg0WwZcAfiqhbQkNooKcllXcnkd OBq0d61yMoYAlSxQTHNc11wbnZg step.md
export_doc step4 实验报告 GnYKwQ63EiWZUjkyeOwcJXvYnRj KVUedo75RoAh7cxgAudcNWPynMb experiment-report.md
export_doc step5 step5 HtXgwUQFGiKrSZkjCt1ckQnZn9f BdSTdAemdot642x9PYic2H0Jnzb step.md
export_doc step5 实验报告 UOaFwX6X7iAQ0Nk1zJ3cdce7nQg LI2pdPfDOovCdrxbfc1cPlAenLb experiment-report.md
