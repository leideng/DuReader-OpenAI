# Prompt Template

The project uses an externalized prompt template that lives at `prompt/template.txt`.

## Current Template

```text title="prompt/template.txt"
请基于给定的文章回答下述问题。
文章:{context}
请基于上述文章回答下面的问题。
问题:{input}
请简短回答。
回答:
```

## Template Variables

The template is formatted with two variables:

- `{context}` - The supporting document context from the dataset
- `{input}` - The user question to answer based on the context

## Why This Format

The template is:

- **Simple**: Direct instruction to answer the question based on the provided article
- **Chinese-native**: Written in natural Chinese for Chinese models
- **Minimal**: No unnecessary instructions that would increase prompt length

## Customization

You can modify `prompt/template.txt` to change the prompting strategy without modifying any evaluation code. All evaluations will automatically use your updated template.
