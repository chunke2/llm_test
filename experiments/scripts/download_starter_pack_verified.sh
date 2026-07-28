#!/usr/bin/env bash
set -euo pipefail

repo_root="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
base_url="https://huggingface.co/datasets/karpathy/llmc-starter-pack/resolve/main"

files=(
  "gpt2_124M.bin|497904640|3da8b207584030bcdcd207cf7a99952e3421dce92da218b351071857511bf162|gpt2_124M.bin"
  "gpt2_124M_bf16.bin|248952832|6661f45628102b4c6e86835d9057b5ba2c024dbf9b81445175e258b7878a1a6f|gpt2_124M_bf16.bin"
  "gpt2_124M_debug_state.bin|549369860|a80c4448eed47c561314a2189e62c49b2c7477c265fc2f15c8470d18401693d9|gpt2_124M_debug_state.bin"
  "gpt2_tokenizer.bin|372108|6f3abc21e444e4e8300e225f4e03da48ea121cf17e30f67009b8dad7a66c2f13|gpt2_tokenizer.bin"
  "tiny_shakespeare_train.bin|611544|8a70606be574040c26d225694f5f9759973b419852d22f7fe5c118e1b359dcc8|dev/data/tinyshakespeare/tiny_shakespeare_train.bin"
  "tiny_shakespeare_val.bin|66560|fe99db720dc7c83e694806d4e047a952909411da1daccde4ccc2e55f40882a62|dev/data/tinyshakespeare/tiny_shakespeare_val.bin"
  "hellaswag_val.bin|3762386|59509ba401e820f2cd8b579141354e8ebc14a20f8e2cc181f1836f1d0e59eb99|dev/data/hellaswag/hellaswag_val.bin"
)

verify_file() {
  local path="$1"
  local expected_size="$2"
  local expected_sha="$3"
  local actual_size
  local actual_sha

  [[ -f "$path" ]] || return 1
  actual_size="$(stat -c '%s' "$path")"
  [[ "$actual_size" == "$expected_size" ]] || return 1
  actual_sha="$(sha256sum "$path" | cut -d ' ' -f 1)"
  [[ "$actual_sha" == "$expected_sha" ]]
}

download_file() {
  local name="$1"
  local expected_size="$2"
  local expected_sha="$3"
  local relative_path="$4"
  local destination="$repo_root/$relative_path"
  local partial="$destination.part"
  local attempt

  mkdir -p "$(dirname "$destination")"
  if verify_file "$destination" "$expected_size" "$expected_sha"; then
    printf 'verified existing %s\n' "$relative_path"
    return
  fi

  if [[ -f "$destination" && ! -f "$partial" ]]; then
    mv "$destination" "$partial"
  fi

  for attempt in 1 2 3; do
    printf 'downloading %s (attempt %d/3)\n' "$relative_path" "$attempt"
    if timeout 900 curl \
      --fail \
      --location \
      --show-error \
      --silent \
      --retry 2 \
      --retry-delay 2 \
      --continue-at - \
      --output "$partial" \
      "$base_url/$name?download=true"; then
      if verify_file "$partial" "$expected_size" "$expected_sha"; then
        mv "$partial" "$destination"
        printf 'verified %s size=%s sha256=%s\n' \
          "$relative_path" "$expected_size" "$expected_sha"
        return
      fi
      printf 'verification failed for %s after attempt %d\n' \
        "$relative_path" "$attempt" >&2
    else
      printf 'download failed for %s after attempt %d\n' \
        "$relative_path" "$attempt" >&2
    fi
  done

  printf 'exhausted retries for %s\n' "$relative_path" >&2
  return 1
}

for entry in "${files[@]}"; do
  IFS='|' read -r name size sha relative_path <<<"$entry"
  download_file "$name" "$size" "$sha" "$relative_path"
done

printf 'starter_pack_verified files=%d\n' "${#files[@]}"
