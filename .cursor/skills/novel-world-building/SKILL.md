---
name: novel-world-building
description: Manage narrative world-building by retrieving location descriptions and setting details from Milvus. Ensures geographical consistency, climate accuracy, architectural details, and cultural specifics remain coherent. Use when introducing new locations, describing familiar settings, or when model needs environmental context to ground story scenes.
---

# Novel World-Building

Ground narrative descriptions by retrieving location and setting details from Milvus before generating environmental content.

## Workflow

Before writing scenes set in specific locations:

1. **Identify primary locations**
   - Extract all location names from the new scene
   - Categorize as: cities, buildings, natural areas, fantasy realms
   - Note if location is first mention or recurring

2. **Query Milvus for location details**
   - For each location, query: `"[location name] description, geography, architecture"`
   - Retrieve top 8-10 chunks for comprehensive context
   - Combine results into a location profile

3. **Extract environmental details**
   From retrieved passages, document:
   - **Geography**: terrain, climate, distance to other locations
   - **Architecture**: buildings, streets, landmarks
   - **Atmosphere**: mood, lighting, sensory details
   - **Inhabitants**: who lives/works there, population characteristics
   - **History**: past events that shaped the location
   - **Unique elements**: distinctive features that make it memorable

4. **Check for consistency**
   - Verify climate aligns with geography
   - Confirm travel times between locations match previous mentions
   - Check that established landmarks remain in same positions
   - Validate population and infrastructure scale

5. **Inject into prompt context**
   - Include location profiles in system prompt before generation
   - Format as structured location data with key characteristics
   - Use specific sensory details to enrich descriptions

6. **Generate environment-grounded scenes**
   - Model writes scenes with retrieved location context
   - Environmental descriptions now consistent with established world
   - New location details embeddable for future retrieval

## Location Profile Template

Use this format when injecting location data:

```
Location: [Location Name]
- Geography: [terrain, climate, cardinal position]
- Architecture: [key buildings, streets, landmarks]
- Atmosphere: [mood, lighting, sensory details]
- Inhabitants: [who's here, population type]
- History: [relevant past events]
- Travel: [distance/time to connected locations]
- Unique elements: [distinctive features]
```

## Query Strategy

**Effective Milvus queries for location retrieval:**

- `"[location] geography, landscape, terrain"` → spatial consistency
- `"[location] architecture, buildings, streets"` → structural details
- `"[location] weather, climate, season"` → environmental conditions
- `"[location] inhabitants, people, culture"` → social context
- `"[location] description, appears, looks"` → sensory grounding
- `"[location] history, past, event"` → contextual depth
- `"travel from [location A] to [location B]"` → geographic relationships

## World-Building Consistency Checks

Prevent environmental contradictions:
- **Geography**: locations can't move; travel times must be consistent
- **Architecture**: buildings once described shouldn't change appearance
- **Climate**: seasonal consistency within timeframe
- **Population**: established city sizes shouldn't drastically change
- **History**: past events shouldn't be retconned
- **Accessibility**: roads/paths established remain accessible
- **Cultural markers**: unique cultural elements stay consistent

## Multi-Location Context

For scenes spanning multiple locations:

1. Query each location separately
2. Retrieve travel/distance information between them
3. Verify timeline feasibility (character can't be in two places simultaneously)
4. Include all location profiles in single context injection

## Tips

- Query before introducing locations or during location changes
- Include surrounding areas/regions for geographic grounding
- When characters travel, retrieve both origin and destination
- Describe weather/season consistently within narrative timeframe
- After generating, embed location descriptions to enrich vector DB
