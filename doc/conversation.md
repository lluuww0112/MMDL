# Conversation Template

평가에서 모델에 전달할 텍스트 프롬프트는 `model/conversation.py`의 **고정 preset**으로 관리합니다. YAML에는 프롬프트 본문을 직접 쓰지 않고 preset 이름만 지정합니다.

```yaml
# config/qwen.yaml
prompt:
  conversation: qwen3_vl_mmmu
```

`eval/qwen/mmmu.py`와 `eval/custom/mmmu.py`는 이 이름으로 preset을 찾은 뒤, MMMU 문항과 이미지를 Qwen chat template에 넣습니다. 등록되지 않은 이름을 설정하면 현재 지원하는 preset 목록과 함께 오류가 발생합니다.

## 입력이 만들어지는 순서

```text
MMMU record (question, options, images)
        │
        ├─ conversation.render(...) → 사용자 텍스트
        ├─ images → 사용자 content의 image 항목들
        └─ Qwen processor.apply_chat_template(...) → 최종 모델 입력
```

이미지는 텍스트보다 앞에 배치됩니다. preset에 `system_prompt`가 있으면 system message가 먼저 추가되고, 그 뒤에 이미지와 렌더링한 텍스트를 가진 user message가 추가됩니다.

## 현재 preset

| 이름 | 클래스 | system prompt | 용도 |
| --- | --- | --- | --- |
| `default` | `Conversation` | 없음 | 별도 서식 없이 fallback prompt를 사용합니다. |
| `qwen3_vl_mmmu` | `MMMUConversation` | 없음 | Qwen3-VL MMMU 평가용 객관식 문항 형식입니다. `config/qwen.yaml`의 기본값입니다. |

### `qwen3_vl_mmmu`

`question`이 비어 있지 않고 `options`가 list/tuple(또는 list literal 문자열)이면 아래 형식으로 렌더링합니다.

```text
Question: {question}
Options:
A. {options[0]}
B. {options[1]}
...
Please select the correct answer from the options above.
```
`options`가 비어 있으면 question만 출력합니다.

```text
Question: {question}
```

### `default`와 fallback

`default` preset은 레코드를 가공하지 않고 평가 코드가 만든 fallback text를 사용합니다. fallback은 객관식일 때 `Question`, `Options`, 선택 안내 문장을 포함하며, 선택지가 없으면 question만 포함합니다.

`qwen3_vl_mmmu`도 `question`이 문자열이 아니거나 빈 문자열이면 fallback으로 돌아갑니다. 따라서 데이터가 불완전해도 평가 코드의 기본 형식으로 시도할 수 있습니다.

## 새 preset 추가

새 template은 YAML에 문자열을 추가하는 것으로 만들 수 없습니다. `model/conversation.py`에 immutable `Conversation` subclass를 정의하고 `CONVERSATION_TEMPLATES`에 이름을 등록해야 합니다.

```python
@dataclass(frozen=True)
class ShortAnswerConversation(Conversation):
    def render(self, document, *, fallback):
        question = document.get("question")
        return f"Answer briefly.\\nQuestion: {question}" if isinstance(question, str) else fallback


CONVERSATION_TEMPLATES["short_answer"] = ShortAnswerConversation(name="short_answer")
```

그 다음 config에서 `prompt.conversation: short_answer`로 선택합니다. `render()`는 항상 문자열을 반환해야 하며, 누락되었거나 예상 밖의 필드를 만났을 때는 `fallback`을 반환하도록 구현하는 편이 안전합니다.

템플릿을 변경하면 실제 입력 문장과 생성 결과가 달라지므로, 기존 실험과 결과를 비교해야 한다면 preset 이름을 새로 만들고 기존 preset은 변경하지 않는 것을 권장합니다.
