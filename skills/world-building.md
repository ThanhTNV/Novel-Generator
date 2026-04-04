# World-Building Skill

Before writing scenes in specific locations:

1. **Retrieve** all stored passages describing the location(s) in the scene.
2. **Extract** from retrieved passages:
   - Geography and climate
   - Architecture and landmarks
   - Atmosphere and sensory details
   - Inhabitants and culture
   - Travel distances to related locations
3. **Verify** consistency: buildings don't move, climates don't shift, distances stay stable.
4. **Inject** location profiles into the generation context.
5. **After generation**, embed new location details for future retrieval.

## Location Profile Format
```
[Location Name]
- Geography: ...
- Architecture: ...
- Atmosphere: ...
- Inhabitants: ...
- Key landmarks: ...
- Travel connections: ...
```
