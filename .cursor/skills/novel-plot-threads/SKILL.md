---
name: novel-plot-threads
description: Track and retrieve unresolved plot threads from Milvus to maintain narrative momentum. Pulls story arcs, character conflicts, mysteries, and plot-critical information to prevent dropped storylines and ensure satisfying payoffs. Use when continuing the story, resolving conflicts, advancing plot arcs, or when model needs to remember what's at stake.
---

# Novel Plot Threads

Maintain narrative coherence by retrieving active plot threads and story arcs from Milvus before generating new chapters.

## Workflow

Before generating new story content:

1. **Catalog active plot threads**
   - List major conflicts currently unresolved
   - Note side quests or subplots in progress
   - Identify mysteries waiting for revelation
   - Track character goals and obstacles

2. **Query Milvus for plot context**
   - For each active thread, query: `"[plot element] unresolved, tension, mystery, conflict"`
   - Retrieve relevant passages showing thread introduction and progression
   - Combine results into plot status document

3. **Extract plot information**
   From retrieved passages, document:
   - **Main conflict**: core story tension and stakes
   - **Character goals**: what characters want and why
   - **Obstacles**: what prevents goal achievement
   - **Plot developments**: recent events advancing the story
   - **Mysteries**: unresolved questions needing answers
   - **Clues planted**: foreshadowing or setup for future payoffs
   - **Timeline**: when events occurred relative to current point

4. **Identify narrative pressure points**
   - Which threads should escalate soon?
   - Which mysteries need hints or revelations?
   - Where is reader engagement highest?
   - Which conflicts are about to resolve?

5. **Inject plot context into prompt**
   - Include active plot threads in system prompt
   - Format as numbered list with current status
   - Highlight threads due for advancement
   - Note clues planted that need payoff

6. **Generate plot-advancing content**
   - Model writes scenes that address active threads
   - Narrative tension maintained through retrieved context
   - Prevents forgotten subplots and dangling threads
   - New plot developments embeddable for continuity

## Plot Thread Tracking Template

Use this format when injecting plot data:

```
Active Plot Threads:

Main Arc:
1. [Thread name]: [current status]
   - Introduced: [when/where]
   - Latest development: [what happened last]
   - Stakes: [why this matters]
   - Next escalation: [what should happen]

Subplot A:
2. [Thread name]: [current status]
   - Key characters involved: [who cares]
   - Unresolved question: [what needs answering]
   - Clues planted: [setup for payoff]
   - Resolution timeline: [when should resolve]

Mystery/Tension:
3. [Thread name]: [current status]
   - What reader doesn't know: [hidden information]
   - When reveal should occur: [narrative timing]
   - How to maintain tension: [sustain reader interest]
```

## Query Strategy

**Effective Milvus queries for plot retrieval:**

- `"[plot element] unresolved, conflict, tension"` → active conflicts
- `"[character] goal, wants, motivation"` → character objectives
- `"[plot element] mystery, secret, revelation"` → mysteries awaiting answers
- `"[plot element] introduced, setup, foreshadowing"` → established expectations
- `"[character A] versus [character B]"` → interpersonal conflicts
- `"[plot element] stakes, consequences, danger"` → narrative pressure
- `"deadline, deadline, must complete"` → time-sensitive plot points

## Plot Thread Maintenance

Track these to prevent narrative failure:

**Dropped threads** - Subplots introduced but never resolved
- Query every 2-3 chapters to check mention status
- If thread absent, either resolve or trigger reminder

**Delayed payoffs** - Setup without sufficient escalation
- Track clues planted; ensure hints don't age too much
- Escalate inactive conflicts gradually

**Contradictory developments** - New plot events that contradict setup
- Retrieve original introduction when advancing threads
- Verify new direction aligns with established context

**Pacing issues** - Too many threads active simultaneously
- Retrieve recent thread activity to gauge saturation
- Ensure adequate focus on primary conflicts

## Multi-Thread Retrieval

For complex scenes affecting multiple plot threads:

1. Query each thread independently
2. Verify no contradictory developments
3. Check timeline feasibility (events must occur in logical order)
4. Include all threads in unified context for coherent scene

## Tips

- Query before major plot developments or turning points
- Retrieve threads before resolution to ensure satisfying payoff
- Include supporting details that flavor and deepen threads
- Track which threads readers are most invested in
- After generating, embed new plot developments to update thread status
- For mysteries: retrieve clues planted to maintain coherent logic chain
