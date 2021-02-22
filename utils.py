arpabet_vowels = ["AA", "AE", "AH", "AO", "AW", "AX", "AXR", "AY", "EH",
                  "ER", "EY", "IH", "IX", "IY", "OW", "OY", "UH", "UW", "UX"]

arpabet_consonants = ["B", "CH", "D", "DH", "DX", "EL", "EM", "EN", "F", "G",
                      "HH", "H", "JH", "K", "K", "L", "M", "N", "NX", "NG", "P",
                      "Q", "R", "S", "SH", "T", "TH", "V", "W", "WH", "Y", "Z", "ZH"]

# Spacy v.2.*.* dependency matcher pattern for getting a sequence 'word' + preposition + 'word' in dependency relations
prep_pattern_spacy2 = [
        {"SPEC": {"NODE_NAME": "fox"}, "PATTERN": {"IS_ALPHA": True}}, 
        {"SPEC": {"NODE_NAME": "prep", "NBOR_NAME": "fox", "NBOR_RELOP": ">"}, "PATTERN": {"TAG":"IN", "DEP": 'prep'}},   # Matches any word > preposition > any word in a dependency relation
        {"SPEC": {"NODE_NAME": "dependent", "NBOR_NAME": "prep", "NBOR_RELOP": ">", }, "PATTERN": {"DEP": "pobj", "IS_ALPHA": True}}
        ]  

uncountables = ["access", "accommodation", "adulthood", "advertising", "advice",
                 "aggression", "aid", "air", "alcohol", "anger", "applause", "arithmetic",
                 "assistance", "athletics", "attention", "baggage", "ballet", "biology", 
                 "blood", "botany", "butter", "carbon", "cardboard", "cash", "chalk", 
                 "chaos", "cheese", "chess", "childhood", "clothing", "coal", 
                 "commerce", "compassion", "comprehension", "corruption", "cotton", 
                 "courage", "damage", "dancing", "danger", "data", "delight", "dignity",
                 "dirt", "dust", "economics", "education", "electricity", "employment",
                 "energy", "engineering", "enjoyment", "entertainment", "envy",
                 "equipment", "ethics", "evidence", "failure", "faith", "fame", 
                 "flour", "flu", "freedom", "fuel", "fun", "furniture",
                 "garbage", "garlic", "gas", "genetics", "gold", "golf", 
                 "gossip", "grass", "grief", "guilt", "gymnastics", "happiness", 
                 "hardware", "harm", "hate", "hatred", "health", "heat", "help",
                 "homework", "honesty", "hospitality", "housework", "humour", "hunger",
                 "hydrogen", "importance", "inflation", "information", "injustice",
                 "innocence", "intelligence", "iron", "irony", "jam", "jealousy", 
                 "jealousy", "judo", "juice", "justice", "karate", "kindness", 
                 "knowledge", "labour", "lack", "laughter", "lava", "leather", "leisure",
                 "lightning", "linguistics", "litter", "livestock", "logic", "loneliness",
                 "love", "luck", "luggage", "machinery", "magic", "mail", "management",
                 "mankind", "marble", "mathematics", "mayonnaise", "measles", "meat",
                 "metal", "methane", "milk", "money", "mud", "music", "news", "nitrogen",
                 "nonsense", "nurture", "nutrition", "obedience", "obesity", "oil",
                 "oxygen", "passion", "patience", "permission", "physics", "poetry",
                 "pollution", "poverty", "pride", "production", "progress", "psychology",
                 "publicity", "punctuation", "quantity", "quartz", "racism", "rain",
                 "recreation", "relaxation", "reliability", "research", "revenge", 
                 "rice", "rubbish", "safety", "salt", "sand", "satire", 
                 "scenery", "seafood", "seaside", "shame", "shopping", "silence", 
                 "sleep", "smoke", "smoking", "soap", "software", "sorrow", "speed",
                 "spelling", "steam", "strength", "stupidity", "success", "sugar",
                 "sunshine", "symmetry"]

uncountable_change = {
        "those": "that",
        "these": "this",
        "many": "much",
        "few": "little"
}