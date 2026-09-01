import spacy
import csv
from wordfreq import word_frequency

class PromptAnalysisResources:
    def __init__(self,
                 spacy_model_name="en_core_web_sm",
                 concreteness_path="concreteness.csv",
                 rareact_pairs_path="rareact_pairs.csv",
                 spatial_terms_path="spatial_terms.txt"):
        self.spacy_model_name = spacy_model_name
        self.concreteness_path = concreteness_path
        self.rareact_pairs_path = rareact_pairs_path
        self.spatial_terms_path = spatial_terms_path

        self._nlp = None
        self._concreteness_dict = None
        self._rareact_verbs = None
        self._spatial_terms = None

    @property
    def nlp(self):
        if self._nlp is None:
            self._nlp = spacy.load(self.spacy_model_name)
        return self._nlp

    @property
    def concreteness_dict(self):
        if self._concreteness_dict is None:
            self._concreteness_dict = self._load_concreteness_dict()
        return self._concreteness_dict

    @property
    def rareact_verbs(self):
        if self._rareact_verbs is None:
            self._rareact_verbs = self._load_rareact_verbs()
        return self._rareact_verbs

    @property
    def spatial_terms(self):
        if self._spatial_terms is None:
            self._spatial_terms = self._load_list(self.spatial_terms_path)
        return self._spatial_terms

    def get_word_frequency(self, word):
        return word_frequency(word.lower(), "en")

    def get_num_objects(self, doc):
        return sum(1 for token in doc if token.dep_ in {"nsubj", "dobj", "pobj"})

    def get_num_verbs(self, doc):
        return sum(1 for token in doc if token.pos_ == "VERB")

    # def get_num_positions(self, prompt):
    #     prompt_lower = prompt.lower()
    #     return sum(1 for phrase in self.spatial_terms if phrase in prompt_lower)

    def get_num_positions(self, doc):
        tokens = [token.text.lower() for token in doc]
        token_str = " ".join(tokens)
    
        count = 0
        for phrase in self.spatial_terms:
            phrase_tokens = phrase.split()
            if len(phrase_tokens) == 1:
                # Single-word: check exact match in tokens
                count += tokens.count(phrase)
            else:
                # Multi-word: sliding window match
                phrase_str = " ".join(phrase_tokens)
                count += token_str.count(phrase_str)
        return count

    def get_num_attributes(self, doc):
        return sum(1 for token in doc if token.pos_ == "ADJ" or token.dep_ in {"amod", "acomp"})

    def get_num_namedEntities(self, doc):
        return sum(1 for ent in doc.ents if ent.label_ in {"PERSON", "ORG", "GPE"})

    def get_unusual_combinations(self, doc):
        # Placeholder for future implementation
        return []

    def extract_features(self, prompt):
        doc = self.nlp(prompt)

        # Prompt length
        prompt_length = sum(1 for token in doc if not token.is_punct)

        # Token rarity
        token_rarity = sum(self.get_word_frequency(token.text) < 1e-5 for token in doc)

        # Number of objects
        num_objects = self.get_num_objects(doc)

        # Attribute density (adjectives + modifiers)
        attribute_density = self.get_num_attributes(doc)

        # Spatial relations (detect both single and multi-word phrases)
        spatial_relations = self.get_num_positions(doc)

        # Action verbs
        action_verbs = self.get_num_verbs(doc)

        # Named entities of type PERSON, ORG, GPE
        named_entities = self.get_num_namedEntities(doc)

        # Abstractness
        abstractness = sum(
            1 for token in doc if self.concreteness_dict.get(token.text.lower(), 5.0) < 2.5
        )

        return {
            "prompt_length": prompt_length,
            "token_rarity": token_rarity,
            "num_objects": num_objects,
            "abstractness": abstractness,
            "attribute_density": attribute_density,
            "spatial_relations": spatial_relations,
            "action_verbs": action_verbs,
            "named_entities": named_entities,
        }

    def _load_concreteness_dict(self):
        scores = {}
        try:
            with open(self.concreteness_path, newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    word = row["Word"].lower()
                    score = float(row["Conc.M"])
                    scores[word] = score
        except FileNotFoundError:
            print(f"Warning: Concreteness file not found at '{self.concreteness_path}'.")
        return scores

    def _load_rareact_verbs(self):
        verbs = set()
        try:
            with open(self.rareact_pairs_path, newline='') as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) >= 2:
                        _, verb = row[0].strip().lower(), row[1].strip().lower()
                        verbs.add(verb)
        except FileNotFoundError:
            print(f"Warning: RareAct pairs file not found at '{self.rareact_pairs_path}'.")
        return verbs

    def _load_list(self, filepath):
        items = set()
        try:
            with open(filepath, 'r') as f:
                for line in f:
                    item = line.strip().lower()
                    if item:
                        items.add(item)
        except FileNotFoundError:
            print(f"Warning: List file not found at '{filepath}'.")
        return items
