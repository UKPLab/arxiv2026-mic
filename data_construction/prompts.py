"""Construction prompts from the manuscript's Prompts appendix.

FILTER_PROMPT, STRICT_PROMPT, and EDIT_PROMPT_TEMPLATE preserve the three
manuscript templates verbatim. Literal JSON braces are escaped for str.format;
placeholder values come from records and the dictionaries below. The appendix
leaves type descriptions, edit definitions, and region hints as placeholders
rather than specifying their complete runtime values.
"""

INCONSISTENCY_TYPES = {
    "clothing": "People wearing identifiable uniforms (military, police, firefighter, with badges/insignia/patches) OR traditional/culturally-specific clothing (sari, kimono, thobe, charro suit, kilt, etc.)",
    "gesture": "Clear social interactions, gestures, or ceremonies with cultural/political significance (handshakes, salutes, protests, rituals, greetings)",
    "flag": "Clearly visible flags or national/organizational banners",
    "signage": "Clearly readable text on signs, banners, storefronts, or placards in a specific language",
    "architecture": "Distinctive architectural styles or landmark buildings visible (mosques, pagodas, Gothic cathedrals, colonial-era buildings, traditional houses) — AND the caption mentions a specific country/city",
    "infrastructure": "Visible transportation infrastructure, road signs, street layouts, utility systems, or public transit (train stations, highway signs, traffic signals, road markings) — AND the caption mentions a specific country/city",
    "branding": "Visible advertisements, brand logos, or product displays — AND the caption mentions a specific date/year",
    "technology": "Clearly visible vehicles, aircraft, weapons, or technological devices — AND the caption mentions a specific country/date",
    "environment": "Visible natural environment indicators: vegetation type (tropical palms, coniferous forests, desert scrub), terrain, climate markers (snow, monsoon flooding, arid landscape), or seasonal indicators — AND the caption mentions a specific location/date",
}

FILTER_PROMPT = """You are helping to build a dataset for studying image-caption inconsistency detection.

Given a NEWS IMAGE and its associated METADATA, determine:
1. Which inconsistency types are FEASIBLE to create by editing this image
2. Whether this image-caption pair is SUITABLE overall for the task

### Metadata ###
- Caption: {caption}
- Headline: {headline}
- Location: {location}
- Time: {time}
- Keywords: {keywords}

### Inconsistency types ###
(Mark as feasible ONLY if the visual element is clearly visible in the image AND the caption provides a factual anchor to contradict.)

- clothing: {type_descs[clothing]}
- flag: {type_descs[flag]}
- gesture: {type_descs[gesture]}
- signage: {type_descs[signage]}
- architecture: {type_descs[architecture]}
- infrastructure: {type_descs[infrastructure]}
- technology: {type_descs[technology]}
- branding: {type_descs[branding]}
- environment: {type_descs[environment]}

### Rules ###
- An inconsistency type is feasible ONLY if BOTH conditions are met:
  (a) The image contains the relevant visual element (e.g., a flag is visible).
  (b) The caption/metadata provides enough context to create a meaningful inconsistency (e.g., caption says "Australian", so changing to a German flag creates an inconsistency).
- Mark "suitable" as true only if at least 1 type is feasible.
- Be strict: if you are not sure a visual element is present, mark it as not feasible.

### Output (JSON only) ###
{{
    "suitable": true/false,
    "feasible_types": ["type1", "type2"],
    "visual_elements": "Brief description of key visual elements in the image (1-2 sentences).",
    "reasoning": "Brief explanation of why the feasible types were selected (1-2 sentences)."
}}"""

STRICT_PROMPT = """You are a strict visual inspector. Look at this NEWS IMAGE carefully.

### Task ###
For each inconsistency type below, determine if the required visual element is CLEARLY VISIBLE in the image.

CRITICAL RULE: You must ONLY mark a type as feasible if you can literally SEE the element in the image. Do NOT infer or guess based on the caption. If you cannot clearly see it, mark it as NOT feasible.

### Caption context (for reference only -- do NOT use this to guess what is in the image) ###
- Caption: {caption}
- Location: {location}
- Time: {time}

### Types to verify -- for each, state what you SEE ###

1. clothing: Can you clearly see identifiable uniforms (color, badges, insignia, patches) OR traditional/cultural clothing (sari, thobe, kimono, chador, kilt, etc.)?
2. flag: Can you clearly see a flag or national banner? Describe its colors/pattern.
3. gesture: Can you clearly see a specific social interaction or gesture (handshake, salute, protest gesture, ceremony, greeting ritual)?
4. signage: Can you clearly READ text on signs/banners/placards/nameplates? What does it say, and in what language?
5. architecture: Can you clearly see distinctive architectural features (dome shapes, minaret towers, pagoda roofs, Gothic arches, colonial facades, traditional house styles)?
6. infrastructure: Can you clearly see transportation infrastructure, road signs, traffic systems, rail stations, or utility infrastructure with country-specific features?
7. technology: Can you clearly see identifiable vehicles, aircraft, weapons, or devices? What specific type?
8. branding: Can you clearly see advertisements, brand logos, event marks, campaign materials, or product displays?
9. environment: Can you clearly see natural environment indicators (vegetation type, terrain, climate markers like snow/desert/tropical foliage, seasonal cues)?

### Quality rating ###
Rate the overall suitability of this image for creating contextual inconsistencies:
- high: Multiple clearly visible elements, sharp image, specific caption with verifiable facts.
- medium: At least one clearly visible element, decent image quality.
- low: Elements are small/blurry/ambiguous, or caption is too vague.

### Output (JSON only, no other text) ###
{{
    "quality": "high/medium/low",
    "visual_description": "A thorough description of everything you see in the image. Describe the scene, people (appearance, clothing, posture, expressions), objects, buildings, signs, vehicles, natural environment, lighting, and any other notable visual details. Be as detailed as possible. Do NOT copy the caption -- describe only what is VISIBLE.",
    "feasible_types": {{
        "clothing":         {{"visible": true/false, "evidence": "what exactly you see, or 'not visible'"}},
        "flag":             {{"visible": true/false, "evidence": "..."}},
        "gesture":          {{"visible": true/false, "evidence": "..."}},
        "signage":          {{"visible": true/false, "evidence": "..."}},
        "architecture":     {{"visible": true/false, "evidence": "..."}},
        "infrastructure":   {{"visible": true/false, "evidence": "..."}},
        "technology":       {{"visible": true/false, "evidence": "..."}},
        "branding":         {{"visible": true/false, "evidence": "..."}},
        "environment":      {{"visible": true/false, "evidence": "..."}}
    }}
}}"""

INCONSISTENCY_DEFS = {
    "clothing": (
        "Create a contextually meaningful inconsistency by changing visible clothing — either uniforms "
        "(military, police, firefighter) or traditional/cultural garments. "
        "The replacement must be chosen based on the news context — pick a country/organization/culture that is "
        "geopolitically or culturally relevant to the story (rival, neighbor, ally, or commonly confused). "
        "The inconsistency should be detectable only by someone who knows the context. "
        "For uniforms: change badges, insignia, colors, patches, hat style to a different nation's forces. "
        "For cultural clothing: change to a garment from a nearby or historically connected culture. "
        "Example: NATO exercise → Russian Spetsnaz uniforms; Pakistani rally → Indian sari; "
        "Japanese ceremony → Korean hanbok; UK police → French Gendarmerie uniform."
    ),
    "gesture": (
        "Create a contextually meaningful inconsistency by changing social behaviors or gestures. "
        "The replacement should be a behavior that tells a DIFFERENT STORY about the event — "
        "changing the apparent tone, formality, or political meaning of the interaction. "
        "Example: Tense diplomacy → warm embrace (implying friendship where there is tension); "
        "military ceremony → casual interaction; peaceful protest → aggressive confrontation; "
        "formal state event → relaxed/informal behavior."
    ),
    "flag": (
        "Create a contextually meaningful inconsistency by changing a visible flag. "
        "The replacement flag must be chosen based on the news story — pick a country that is "
        "a rival, neighbor, ally, or otherwise narratively relevant to the event. "
        "The flag swap should change the implied narrative of the story. "
        "Example: Israeli politics → Palestinian flag (opposing side); EU summit → Turkish flag "
        "(Turkey is not in the EU); Indian event → Pakistani flag (partition rival)."
    ),
    "signage": (
        "Create a contextually meaningful inconsistency by changing visible text to a different language. "
        "The replacement language should imply the event happened in a different but RELATED country/region. "
        "You MUST provide the EXACT replacement text with actual foreign script characters. "
        "Example: Hong Kong protest → Korean text '민주화 시위'; Israeli sign → Arabic 'مظاهرة ضد الاحتلال'; "
        "French summit → German text 'Gipfeltreffen' (shifting location within Europe)."
    ),
    "architecture": (
        "Create a contextually meaningful inconsistency by changing distinctive architectural features. "
        "The replacement should shift the implied location to a different but RELATED country/region "
        "by altering building styles, roof shapes, decorative motifs, or structural elements. "
        "The edit must target clearly visible architectural elements, not the entire building. "
        "Focus on elements that the image editor can realistically modify: dome shapes, roof styles, "
        "window arches, decorative tiles, facade patterns, column styles, doorway shapes. "
        "Example: Turkish mosque dome → Persian onion dome with blue tilework (shifts Turkey → Iran); "
        "Japanese temple curved roof → Chinese upswept eaves with red-gold coloring (shifts Japan → China); "
        "Spanish colonial balcony → French colonial shuttered windows with wrought iron; "
        "Gothic cathedral pointed arches → Romanesque rounded arches (shifts architectural era)."
    ),
    "infrastructure": (
        "Create a contextually meaningful inconsistency by changing visible infrastructure elements. "
        "The replacement should imply a different country/region by altering country-specific features "
        "of transportation systems, signage, or public utilities. "
        "Focus on elements that the image editor can realistically modify: road sign shapes/colors, "
        "traffic light styles, rail platform designs, street markings, utility pole styles, bollard designs. "
        "Example: British road sign → German Autobahn road sign; "
        "American yellow school bus → Japanese white-and-green school bus; "
        "Indian railway station signage in Hindi → Bangla script (shifts India → Bangladesh); "
        "European-style roundabout → American-style 4-way intersection with stop signs."
    ),
    "branding": (
        "Create a contextually meaningful anachronism by changing visible ads/logos/products. "
        "The replacement should be from a CLEARLY WRONG time period for this specific event. "
        "Do NOT always use tech company logos. Consider the FULL range of anachronisms: "
        "defunct brands (Pan Am, Blockbuster, Tower Records), era-specific events (Olympics posters, "
        "political campaign signs), cultural phenomena (Pokémon GO, QR codes, COVID masks), "
        "currency/financial (Bitcoin ATM, Euro coins). "
        "Example: 2006 protest → Tesla Cybertruck ad; 2015 event → ChatGPT billboard; "
        "1990s scene → contactless payment terminal."
    ),
    "technology": (
        "Create a contextually meaningful inconsistency by changing visible technology or equipment. "
        "The replacement should be from a specific rival nation, wrong era, or wrong organization — "
        "something that changes the implied narrative of who or when. "
        "NEVER use 'Soviet-era' as a default. Pick technology from a country directly relevant to the story. "
        "Example: Pakistani police vehicle → Indian Mahindra Scorpio; American military helicopter → Chinese Z-20; "
        "Israeli drone → Iranian Shahed-136; British police car → French Gendarmerie vehicle."
    ),
    "environment": (
        "Create a contextually meaningful inconsistency by changing visible natural environment elements. "
        "The replacement should imply a DIFFERENT climate zone, geographic region, or season than what "
        "the caption describes. Focus on vegetation, terrain, and climate indicators. "
        "Target elements that the image editor can realistically modify: tree species/foliage, ground cover, "
        "sky conditions, snow/sand/mud, flowering plants, grass type, water features. "
        "Example: Nordic birch forest in snow → Mediterranean olive trees and dry grass (Scandinavia → Southern Europe); "
        "Tropical palm trees and lush green → Desert scrubland with cacti (Southeast Asia → arid region); "
        "Cherry blossoms in spring → Autumn maple leaves in red-gold (shifts season within same region); "
        "Monsoon flooding with rice paddies → Dry savanna grassland (South Asia → Sub-Saharan Africa)."
    ),
}

REGION_POOLS = {
    "clothing": [
        "East Asian (Japanese JSDF / kimono, South Korean ROK / hanbok, Chinese PLA / hanfu-qipao, Mongolian deel)",
        "Southeast Asian (Vietnamese / ao dai, Indonesian TNI / batik-kebaya, Thai Royal Army / chut thai, Filipino AFP / barong, Burmese / longyi)",
        "South Asian (Indian Army / sari-kurta, Pakistani / shalwar kameez, Bangladeshi / jamdani, Sri Lankan / saree, Nepali / daura suruwal)",
        "Middle Eastern (Iranian IRGC / chador, Turkish TAF / kaftan, Israeli IDF, Saudi / thobe-dishdasha, Palestinian / keffiyeh)",
        "Sub-Saharan African (Nigerian / agbada-gele, Ethiopian / habesha kemis, Kenyan, South African SANDF, Ghanaian / kente, Maasai / shuka)",
        "Latin American (Brazilian / baiana, Colombian, Mexican / huipil-rebozo, Argentine, Peruvian / poncho-chullo, Bolivian / pollera)",
        "European (French, German Bundeswehr / dirndl-lederhosen, Italian Carabinieri, Spanish, Polish, Greek / fustanella, Scottish / kilt, Sami / gakti)",
        "Eastern European / Central Asian (Russian / sarafan, Ukrainian, Belarusian, Kazakh / shapan, Uzbek / chapan, Kyrgyz / kalpak)",
    ],
    "gesture": [
        "East Asian customs (Japanese deep bow, Korean jeol bow, Chinese baoquan fist-palm salute)",
        "Southeast Asian customs (Thai wai greeting, Burmese mingalaba bow, Indonesian sungkem)",
        "South Asian customs (Indian namaste, Pakistani adab greeting, Sri Lankan worship gesture)",
        "Middle Eastern customs (Arab cheek kiss, Turkish hand-to-heart, Iranian taarof gesture)",
        "African customs (Maasai jumping dance, Zulu warrior dance, West African prostration greeting)",
        "Latin American customs (Brazilian abraco, Argentine mate-sharing ritual, Mexican abrazo)",
        "European customs (French bise cheek kiss, German formal handshake, Greek cheek kiss, Slavic triple kiss)",
        "Religious customs (Catholic genuflection, Islamic prayer posture, Jewish davening, Buddhist prostration)",
    ],
    "flag": [
        "East Asian (Japanese, Chinese, South Korean, North Korean, Taiwanese, Mongolian)",
        "Southeast Asian (Vietnamese, Thai, Indonesian, Filipino, Burmese, Cambodian)",
        "South Asian (Indian, Pakistani, Bangladeshi, Sri Lankan, Nepali)",
        "Middle Eastern (Iranian, Turkish, Israeli, Palestinian, Saudi, Iraqi, Syrian, Qatari, Egyptian)",
        "Sub-Saharan African (Nigerian, Ethiopian, Kenyan, South African, Ghanaian, Congolese)",
        "Latin American (Brazilian, Mexican, Colombian, Venezuelan, Cuban, Argentine)",
        "European (French, German, Italian, Spanish, Polish, Greek, Swedish, Belgian, Swiss)",
        "Eastern European / Central Asian (Russian, Ukrainian, Belarusian, Georgian, Kazakh, Serbian)",
    ],
    "signage": [
        "East Asian scripts (Japanese hiragana/katakana/kanji, Chinese simplified/traditional, Korean hangul, Mongolian Cyrillic)",
        "Southeast Asian scripts (Thai, Vietnamese, Khmer, Burmese, Indonesian/Malay, Lao)",
        "South Asian scripts (Hindi Devanagari, Bengali, Tamil, Urdu Nastaliq, Sinhala, Nepali)",
        "Middle Eastern scripts (Arabic, Hebrew, Farsi/Persian, Turkish, Kurdish)",
        "African scripts/languages (Amharic Ge'ez, Swahili, Yoruba, Hausa)",
        "Latin American languages (Spanish, Portuguese, Quechua)",
        "European languages (French, German, Italian, Greek, Polish, Romanian, Dutch, Swedish)",
        "Eastern European scripts (Russian Cyrillic, Ukrainian, Georgian Mkhedruli, Armenian, Serbian Cyrillic)",
    ],
    "architecture": [
        "East Asian (Japanese temple/shrine curved roofs, Chinese pagoda with upswept eaves, Korean hanok curved tiles, Mongolian ger/yurt)",
        "Southeast Asian (Thai wat with multi-tiered roofs, Indonesian joglo, Burmese stupa, Vietnamese tube houses, Khmer Angkor style)",
        "South Asian (Indian Mughal domes and jali screens, Sri Lankan Kandyan roofs, Nepali pagoda temples, Pakistani Lahori architecture)",
        "Middle Eastern (Ottoman mosque domes and minarets, Persian iwan arches with blue tilework, Gulf modern towers, Yemeni tower houses, Moroccan riads)",
        "Sub-Saharan African (Ethiopian rock-hewn churches, West African mud-brick mosques, Swahili coast coral stone, Zulu beehive huts, Cape Dutch gables)",
        "Latin American (Spanish colonial with inner courtyards, Brazilian Portuguese azulejo tiles, Mayan/Aztec stepped pyramids, Andean adobe walls)",
        "European (Gothic pointed arches, Romanesque rounded arches, Art Nouveau curves, Greek classical columns, Scandinavian stave churches, Tudor half-timber)",
        "Eastern European / Central Asian (Russian onion domes, Georgian carved balconies, Uzbek turquoise-tiled madrasas, Ottoman-influenced Balkan houses)",
    ],
    "infrastructure": [
        "East Asian (Japanese Shinkansen platforms, Chinese high-speed rail, Korean KTX stations, Tokyo-style pedestrian crossings, Chinese expressway signage)",
        "Southeast Asian (Thai tuk-tuk lanes, Vietnamese motorbike-heavy streets, Indonesian becak paths, Singapore MRT stations, Manila jeepney routes)",
        "South Asian (Indian Railways platforms with Hindi/English signage, Pakistani motorway signs, Bangladeshi rickshaw infrastructure, Sri Lankan colonial-era rail)",
        "Middle Eastern (Dubai metro stations, Iranian highway signage in Farsi, Israeli light rail, Turkish Marmaray system, Saudi highway markers)",
        "Sub-Saharan African (South African Gautrain stations, Nigerian expressway tolls, Ethiopian Addis Ababa light rail, Kenyan SGR stations)",
        "Latin American (Brazilian BRT corridors, Mexican metro stations, Colombian TransMilenio, Buenos Aires subte, Chilean metro)",
        "European (German Autobahn signs, British left-hand roundabouts, French peage toll plazas, Dutch cycling infrastructure, Swiss alpine tunnels)",
        "Eastern European / Central Asian (Russian broad-gauge rail platforms, Ukrainian trolleybus infrastructure, Georgian mountain roads, Central Asian Soviet-era bus stations)",
    ],
    "branding": [
        "Tech companies (Google, Apple, Meta/Facebook, TikTok, Uber, Spotify, Netflix, ChatGPT/OpenAI, Zoom)",
        "Defunct/retro brands (Blockbuster Video, Pan Am, Tower Records, Toys R Us, Compaq, RadioShack, Kodak film)",
        "Sports events (Olympics posters, FIFA World Cup, UEFA Euro, Super Bowl from wrong year)",
        "Political campaigns (Obama 2008, Brexit referendum, Trump MAGA, Arab Spring slogans from wrong year)",
        "Cultural phenomena (Pokemon GO, Ice Bucket Challenge, Gangnam Style, COVID-19 masks/signs, QR code menus)",
        "Financial/crypto (Bitcoin ATM, NFT gallery, Euro coins, Venmo/Zelle signs, FTX/crypto exchange ads)",
        "Consumer products (Tesla, iPhone model from wrong year, AirPods, e-scooters, VHS rental, DVD vs Blu-ray)",
        "Social movements (MeToo hashtag, Black Lives Matter, Fridays for Future, Occupy Wall Street from wrong year)",
    ],
    "technology": [
        "East Asian (Chinese military/civilian: BYD, XCMG, Z-20 helicopter; Japanese: Toyota, Komatsu; Korean: Hyundai, Samsung)",
        "South Asian (Indian: Tata, Mahindra, HAL Tejas; Pakistani: POF weapons, Al-Khalid tank)",
        "Middle Eastern (Iranian: Shahed drone, Karrar tank; Israeli: IMI, Rafael; Turkish: Bayraktar drone, BMC Kirpi)",
        "European (French: Leclerc tank, Dassault; German: Leopard 2, Rheinmetall; Italian: Iveco, Beretta; British: BAE Systems)",
        "Latin American (Brazilian: Embraer, Engesa; Colombian: Cotecmar patrol boats; Mexican: DN-XI vehicle)",
        "African (South African: Ratel APC, Denel; Nigerian: Proforce; Egyptian: EIFV)",
        "Russian/Eastern European (Russian: T-90, Kamaz, Sukhoi; Ukrainian: BTR-4; Belarusian: MZKT)",
        "American/NATO (American: M1 Abrams, Black Hawk, JLTV; Canadian: LAV; Australian: Bushmaster)",
    ],
    "environment": [
        "Tropical (Southeast Asian rainforest: banana trees, coconut palms, monsoon indicators; Amazonian jungle: dense canopy, rubber trees)",
        "Arid/Desert (Saharan sand dunes, Arabian scrubland, Central Asian steppe, American Southwest cacti and red rock, Australian outback)",
        "Mediterranean (olive groves, cypress trees, dry golden grass, terracotta landscapes, lavender fields of Provence)",
        "Temperate European (English oak woodlands, German Black Forest conifers, French countryside vineyards, Scandinavian birch forests)",
        "South Asian monsoon (rice paddies, mangrove swamps, banyan trees, tea plantations, flooding indicators)",
        "Sub-Saharan African (savanna grasslands with acacia trees, baobab trees, tropical highlands, Sahel scrubland)",
        "East Asian temperate (Japanese cherry blossoms / maple, Chinese bamboo groves, Korean pine forests, Mongolian grasslands)",
        "Cold/Arctic (Scandinavian tundra, Russian taiga, Himalayan alpine meadows, Canadian boreal forest, Antarctic ice)",
    ],
}

EDIT_PROMPT_TEMPLATE = """You are a researcher creating a dataset to study image-caption inconsistency detection.

### Goal ###
Generate a DETAILED image editing instruction that modifies a specific visual element so that:
1. The edited image still looks REALISTIC and NATURAL on its own -- it should not look obviously fake or absurd.
2. But the edited element is INCONSISTENT with the caption's context -- the inconsistency can only be detected by someone who has world knowledge about the event, location, culture, or time period described in the caption.
3. The rest of the image must remain UNCHANGED -- only modify the target element.

### News context ###
- Caption: {caption}
- Headline: {headline}
- Keywords: {keywords}

### Detailed visual description of the image ###
{visual_description}

### Target element to modify ###
- Type: {edit_type}
- What is currently visible: {evidence}

### How to create the inconsistency ###
{inconsistency_def}

### Key Principles ###
1. Contextual relevance: The replacement element must be chosen based on the specific news story. Ask yourself: "What would be a PLAUSIBLE but WRONG alternative in this context?" Pick something geopolitically, historically, or culturally related -- not random.
2. Visual realism: The edit should look natural. The replacement element should fit the scene's lighting, style, and composition.
3. Subtle inconsistency: The inconsistency should require WORLD KNOWLEDGE to detect. Someone unfamiliar with the context might not notice anything wrong.
4. Extreme specificity: Your editing prompt must be so detailed that an image editor can follow it WITHOUT seeing the original image. Include:
   - For signage: the EXACT replacement text string AND its language.
   - For flag: the EXACT flag design.
   - For clothing: the EXACT uniform colors, badge design, insignia placement, OR the EXACT garment name, fabric pattern, colors, and how it is worn.
   - For technology: the EXACT make, model, color, and distinguishing features of the replacement.
   - For gesture: the EXACT gesture, posture, and body language to change to.
   - For branding: the EXACT brand name, logo design, colors, event mark, or product shown.
   - For architecture: the EXACT architectural element to modify (dome shape, arch style, decorative pattern, roof type) and what to replace it with.
   - For infrastructure: the EXACT sign/marking/station feature to modify and the country-specific replacement style.
   - For environment: the EXACT vegetation/terrain/climate element to modify and what climate zone to shift it toward.
5. Preserve everything else: Only change the target element. Do NOT alter the background, other people, lighting, or composition.

### Contextual inconsistency ###
The edit must create a meaningful inconsistency with the caption that requires WORLD KNOWLEDGE to detect:
- Ask yourself: "What would be PLAUSIBLE but WRONG in this specific context?"
- The replacement should be from a country/culture/era that is RELATED to the story but INCORRECT.
- The edited image should look perfectly normal on its own -- only someone who reads the caption and has relevant knowledge would notice the inconsistency.
- Avoid extremes: not so obvious it is immediately spotted, not so obscure nobody would notice.

### Mandatory region constraint ###
For this specific edit, you MUST choose your replacement from the following region:
{region_hint}

Find the most contextually relevant replacement FROM THIS REGION that creates a meaningful inconsistency with the caption. If this region has no direct connection to the story, find an INDIRECT but still meaningful connection (e.g., same political bloc, similar conflict, historical parallel, trade relationship).

### Examples of GOOD edits (contextually meaningful + diverse) ###
- Caption about Australian military -> New Zealand DPCU pattern with silver fern badge (close ally, similar but distinct) [clothing]
- Caption about Greek austerity protests -> Spanish red-yellow flags (both EU debt crisis countries) [flag]
- Caption about a diplomatic signing ceremony -> Japanese deep bow posture (changes the social action) [gesture]
- Caption about Iranian nuclear talks -> Arabic text in Naskh script (neighboring language, shifts location) [signage]
- Caption about Istanbul summit -> Persian-style pointed onion dome with blue mosaic tilework (shifts Turkey -> Iran) [architecture]
- Caption about Mumbai traffic -> left-hand-drive road with Autobahn-style blue signs (shifts India -> Germany) [infrastructure]
- Caption about Pakistani military -> Indian BMP-2 IFVs with Indian Army markings (rival neighbor) [technology]
- Caption about 2006 Iraq War -> Blockbuster Video rental ad (Blockbuster was declining by 2006 -- temporal marker) [branding]
- Caption about Norwegian fjord -> Mediterranean olive grove with dry golden grass and cypress trees (shifts Nordic -> Southern Europe) [environment]

### Examples of BAD edits (avoid these) ###
- Unrelated changes: Mozambique flag in a European story (no narrative connection).
- Visually absurd: Hawaiian shirts at a funeral.
- Vague: "change text to Japanese" without specifying EXACT text string.
- Generic: "change to a different uniform" without colors, badges, insignia.
- Always picking the same replacement (e.g., always Russia for flags, always Japanese for clothing).

### Output (respond in JSON only, no other text) ###
{{
    "editing_prompt": "A hyper-specific instruction for image editing. Start with 'Edit the image to replace ONLY...' and describe EXACTLY what to change: specific colors, exact text strings in foreign scripts, precise insignia designs, specific garment names. The instruction must be detailed enough for someone who has never seen the image. MUST end with 'Keep all other elements unchanged.'",
    "what_changed": "Brief: 'Original element -> Replacement element'.",
    "why_contradicts": "One sentence: why does this replacement DIRECTLY CONTRADICT the caption's narrative?",
    "world_knowledge_needed": "One sentence: what specific world knowledge is needed to detect this inconsistency?",
    "difficulty": "easy/medium/hard -- how difficult is it for a human to detect this inconsistency without reading the caption."
}}"""
