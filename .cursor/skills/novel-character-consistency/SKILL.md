---
name: novel-character-consistency
description: Retrieve and maintain character consistency in novel generation by querying Milvus for all passages mentioning a character. Ensures character traits, appearance, relationships, and voice remain consistent across thousands of tokens. Use when continuing a story, introducing recurring characters, or when the model generates new scenes involving existing characters.
---

# Novel Character Consistency

Maintain accurate character representation by retrieving relevant passages from Milvus before generating new content.

## Workflow

Before writing new character scenes:

1. **Identify characters in the prompt**
   - List all characters that will appear in the new section
   - Note primary vs. secondary characters

2. **Query Milvus for character passages**
   - For each character, query: `"[character name] appearance, personality, relationships"`
   - Retrieve top 5-10 most relevant chunks
   - Combine results into a character profile

3. **Extract key details**
   From retrieved passages, document:
   - **Physical description**: eye color, height, distinguishing features
   - **Personality traits**: temperament, speech patterns, quirks
   - **Relationships**: how they relate to other characters
   - **Role in plot**: their motivation and story arc status

4. **Inject into prompt context**
   - Include extracted details in the system prompt
   - Format as: "Character: [name]\n- Appearance: ...\n- Personality: ...\n- Relationships: ..."
   - Place character profiles in explicit context section before generating

5. **Generate with grounding**
   - Model now writes new scenes with retrieved context
   - Consistency automatically enforced through prompt injection
   - New passages automatically embeddable into Milvus for future retrieval

## Character Profile Template

Use this format when injecting character data into prompts:

```
Character: [Character Name]
- Appearance: [physical traits from retrieved passages]
- Personality: [temperament and speech patterns]
- Relationships: 
  - [Character A]: [relationship type and status]
  - [Character B]: [relationship type and status]
- Current arc: [their unresolved story threads]
- Contradictions to avoid: [traits that shouldn't change]
```

## Query Strategy

**Effective Milvus queries for character retrieval:**

- `"[character] appearance, eyes, hair, clothing"` → physical consistency
- `"[character] said, spoke, thought"` → voice and dialogue patterns
- `"[character] relationship with [other character]"` → interpersonal dynamics
- `"[character] backstory, origin, history"` → motivations and depth
- `"[character] personality, emotional, reaction"` → behavioral consistency

## Common Contradictions to Prevent

Track and verify these don't change mid-narrative:
- Physical traits (eye color, scars, distinguishing marks)
- Speech patterns and dialect
- Core personality traits (temperament, values)
- Relationship status with other characters
- Character's death/absence (major plot points)
- Skills and abilities established earlier

## Tips

- Query before each major character appearance
- Include secondary characters mentioned peripherally
- When characters interact, retrieve profiles for all involved
- After generating, embed new passages to update character knowledge in vector DB
