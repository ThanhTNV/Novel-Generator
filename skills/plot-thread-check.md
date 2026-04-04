# Plot Thread Tracking Skill

Before generating new story content:

1. **Catalog** all active plot threads from retrieved context:
   - Main conflict and current stakes
   - Active subplots and their status
   - Unresolved mysteries and planted clues
   - Character goals and obstacles
2. **Identify** which threads should advance in this chapter.
3. **Check** for threads that have been dormant too long (risk of being dropped).
4. **Inject** active thread summaries into the generation context.
5. **After generation**, verify at least one thread has meaningfully advanced.

## Thread Status Format
```
Thread: [name]
- Status: active | escalating | near-resolution | dormant
- Introduced: chapter [N]
- Last advanced: chapter [N]
- Stakes: ...
- Next beat: ...
```
