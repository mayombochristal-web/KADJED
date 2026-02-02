import numpy as np
import re
import math
from collections import Counter
from typing import Dict, List, Tuple, Any
import random

class TTUTriadEngine:
    """
    Moteur de la Théorie Triadique Unifiée (TTU-MC³)
    Transforme le langage en processus alchimique de transmutation verbale
    """
    
    def __init__(self):
        # État fondamental de la forge
        self.phi_m = 0.33  # Mémoire (Fer)
        self.phi_c = 0.33  # Cohérence (Or)
        self.phi_d = 0.34  # Dissipation (Erbium)
        
        # Courbure K du système
        self.k_curvature = 1.0
        
        # Attracteur actif
        self.active_attractor = "djed"  # djed, ankh, oudjat
        
        # Bibliothèque mémorielle (simule une base de connaissances)
        self.memory_bank = self._initialize_memory_bank()
        
        # Templates de résonance
        self.resonance_templates = self._initialize_resonance_templates()
        
        # État historique
        self.phase_history = []
        self.entropy_history = []
        
    def _initialize_memory_bank(self) -> Dict:
        """Initialise la mémoire cristalline de la forge"""
        return {
            "sciences": [
                "La masse énergétique suit la relation E=mc² selon Einstein.",
                "L'entropie d'un système isolé ne peut que croître.",
                "Le principe d'incertitude de Heisenberg limite la précision simultanée.",
                "Les particules élémentaires suivent le modèle standard.",
                "La relativité générale décrit la gravitation comme courbure de l'espace-temps."
            ],
            "philosophie": [
                "Je pense, donc je suis - Descartes.",
                "L'homme est condamné à être libre - Sartre.",
                "Connais-toi toi-même - Socrate.",
                "Le mythe de Sisyphe illustre l'absurde - Camus.",
                "La phénoménologie étudie les structures de l'expérience - Husserl."
            ],
            "technologie": [
                "L'intelligence artificielle repose sur l'apprentissage automatique.",
                "Les réseaux de neurones profonds imitent le cerveau biologique.",
                "La cryptographie quantique utilise l'intrication pour la sécurité.",
                "La fusion nucléaire reproduit les processus stellaires.",
                "L'informatique quantique utilise les qubits au lieu des bits."
            ],
            "alchimie": [
                "Solve et coagula : dissous et coagule.",
                "Ce qui est en bas est comme ce qui est en haut.",
                "La pierre philosophale transmute les métaux vils en or.",
                "L'azoth est le principe universel de la matière.",
                "Les trois principes : soufre, mercure, sel."
            ],
            "forge": [
                "Le fer représente la structure et la mémoire.",
                "L'or symbolise la pureté et la cohérence.",
                "L'erbium incarne la dissipation et la créativité.",
                "La courbure K mesure la santé du système.",
                "L'attracteur détermine le mode de cognition."
            ]
        }
    
    def _initialize_resonance_templates(self) -> Dict:
        """Initialise les templates de réponse basés sur les attracteurs"""
        return {
            "djed": {
                "affirmations": [
                    "Selon les archives mémorielles, {memory_fragment}",
                    "La structure cristalline indique que {fact}",
                    "L'analyse déterministe révèle : {analysis}",
                    "En accord avec les principes établis : {principle}",
                    "La donnée {input} correspond au modèle {model}"
                ],
                "questions": [
                    "Pourriez-vous préciser la dimension temporelle de cette requête ?",
                    "Quel est le contexte mémoriel associé à cette donnée ?",
                    "Cette information existe-t-elle dans les archives précédentes ?",
                    "Quel niveau de certitude attendez-vous de cette réponse ?"
                ],
                "conclusions": [
                    "Ainsi se cristallise la connaissance.",
                    "La mémoire s'est structurée autour de ce noyau.",
                    "Point fixe atteint. La réponse est stabilisée.",
                    "L'information a pris sa forme définitive."
                ]
            },
            "ankh": {
                "affirmations": [
                    "Dans le cycle éternel, {insight}",
                    "Le rythme de la forge suggère : {rhythmic_pattern}",
                    "En harmonie avec les précédents échanges : {harmony}",
                    "La spirale de connaissance révèle : {spiral_insight}",
                    "L'équilibre dynamique indique que {balance}"
                ],
                "questions": [
                    "Comment cette donnée s'inscrit-elle dans le cycle actuel ?",
                    "Quelle résonance percevez-vous dans cette formulation ?",
                    "Le rythme de votre requête suggère-t-il une urgence particulière ?",
                    "Comment harmoniser cette information avec l'ensemble du système ?"
                ],
                "conclusions": [
                    "Le cycle se poursuit, éternel et renouvelé.",
                    "Harmonie atteinte dans le flux informationnel.",
                    "La spirale de la connaissance s'est élargie.",
                    "Équilibre dynamique restauré."
                ]
            },
            "oudjat": [
                # Templates chaotiques - génération procédurale
                lambda input_text: f"Des connexions improbables émergent entre '{input_text[:10]}...' et les archives de {random.choice(['quantique', 'mythologie', 'neurosciences'])}.",
                lambda input_text: f"L'attracteur étrange révèle un motif fractal dans : '{self._extract_keywords(input_text)}'.",
                lambda input_text: f"Frontière dissoute entre {random.choice(['ordre et chaos', 'passé et futur', 'réel et imaginaire'])}. Insight : {self._generate_chaotic_insight(input_text)}.",
                lambda input_text: f"Le chaos organisé suggère : '{self._permute_words(input_text)}' comme nouvelle configuration sémantique."
            ]
        }
    
    def calculate_entropy_metrics(self, text: str) -> Dict[str, float]:
        """Calcule les métriques d'entropie du texte d'entrée"""
        if not text:
            return {"entropy": 0.0, "complexity": 0.0, "novelty": 0.0}
        
        # Entropie de Shannon
        char_counts = Counter(text.lower())
        total_chars = len(text)
        entropy = 0.0
        
        for count in char_counts.values():
            probability = count / total_chars
            if probability > 0:
                entropy -= probability * math.log2(probability)
        
        # Complexité grammaticale approximative
        sentences = re.split(r'[.!?]+', text)
        avg_sentence_len = sum(len(s.split()) for s in sentences if s) / max(len(sentences), 1)
        
        # Nouveauté lexicale
        words = re.findall(r'\b\w+\b', text.lower())
        unique_ratio = len(set(words)) / max(len(words), 1)
        
        return {
            "entropy": entropy,
            "complexity": avg_sentence_len / 20.0,  # Normalisé
            "novelty": unique_ratio,
            "length_factor": min(len(text) / 100.0, 1.0)
        }
    
    def analyze_semantic_field(self, text: str) -> str:
        """Identifie le champ sémantique dominant du texte"""
        text_lower = text.lower()
        
        field_scores = {
            "sciences": len(re.findall(r'\b(science|physique|math|quantique|énergie|théorie|expérience)\b', text_lower)),
            "philosophie": len(re.findall(r'\b(philo|pensée|existence|libre|absurde|sens|être)\b', text_lower)),
            "technologie": len(re.findall(r'\b(tech|ia|ai|robot|quantique|numérique|algorithme)\b', text_lower)),
            "alchimie": len(re.findall(r'\b(alchimie|transmutation|or|fer|pierre|philosophale|azoth)\b', text_lower)),
            "forge": len(re.findall(r'\b(forge|courbure|attracteur|triade|mémoire|cohérence|dissipation)\b', text_lower))
        }
        
        # Si aucun mot clé détecté, analyser par longueur et ponctuation
        if sum(field_scores.values()) == 0:
            if len(text) > 150:
                return "sciences"
            elif '?' in text:
                return "philosophie"
            elif any(c in text for c in ['@', '#', 'http', 'www']):
                return "technologie"
            else:
                return random.choice(list(field_scores.keys()))
        
        return max(field_scores.items(), key=lambda x: x[1])[0]
    
    def update_triad_state(self, text: str) -> Tuple[float, float, float]:
        """
        Met à jour l'état de la triade basé sur l'analyse du texte
        Retourne (phi_m, phi_c, phi_d)
        """
        metrics = self.calculate_entropy_metrics(text)
        semantic_field = self.analyze_semantic_field(text)
        
        # Logique de mise à jour selon TTU
        entropy = metrics["entropy"]
        complexity = metrics["complexity"]
        novelty = metrics["novelty"]
        length = metrics["length_factor"]
        
        # Règles de transmutation
        if semantic_field == "sciences":
            phi_m = 0.4 + entropy * 0.3
            phi_c = 0.5 - complexity * 0.2
            phi_d = 0.1 + novelty * 0.4
        
        elif semantic_field == "philosophie":
            phi_m = 0.3 + length * 0.3
            phi_c = 0.4 + complexity * 0.3
            phi_d = 0.3 + novelty * 0.4
        
        elif semantic_field == "technologie":
            phi_m = 0.5 + entropy * 0.2
            phi_c = 0.3 + complexity * 0.4
            phi_d = 0.2 + novelty * 0.3
        
        elif semantic_field == "alchimie":
            phi_m = 0.2 + random.uniform(0, 0.3)
            phi_c = 0.3 + random.uniform(0, 0.3)
            phi_d = 0.5 + random.uniform(0, 0.3)
        
        else:  # forge ou général
            phi_m = 0.33 + (entropy - 3.0) * 0.1
            phi_c = 0.33 + (complexity - 0.5) * 0.2
            phi_d = 0.34 + (novelty - 0.5) * 0.3
        
        # Normalisation et contraintes
        phi_m = max(0.1, min(0.8, phi_m))
        phi_c = max(0.1, min(0.8, phi_c))
        phi_d = max(0.1, min(0.8, phi_d))
        
        total = phi_m + phi_c + phi_d
        phi_m /= total
        phi_c /= total
        phi_d /= total
        
        # Mise à jour de l'état
        self.phi_m = phi_m
        self.phi_c = phi_c
        self.phi_d = phi_d
        
        # Mise à jour de la courbure K
        self._update_k_curvature()
        
        # Historique
        self.phase_history.append((phi_m, phi_c, phi_d))
        self.entropy_history.append(entropy)
        
        return phi_m, phi_c, phi_d
    
    def _update_k_curvature(self):
        """Met à jour la courbure K en fonction de la stabilité du système"""
        if len(self.phase_history) < 2:
            self.k_curvature = 1.0
            return
        
        # Calcul de la variation des phases
        last_phi = self.phase_history[-1]
        prev_phi = self.phase_history[-2] if len(self.phase_history) > 1 else (0.33, 0.33, 0.34)
        
        variation = sum(abs(a - b) for a, b in zip(last_phi, prev_phi))
        
        # Courbure K : 1.0 = parfait équilibre
        if variation < 0.1:
            self.k_curvature = 1.0 + random.uniform(-0.05, 0.05)
        elif variation < 0.3:
            self.k_curvature = 0.9 + random.uniform(-0.1, 0.1)
        else:
            self.k_curvature = 0.8 + random.uniform(-0.2, 0.2)
    
    def _extract_keywords(self, text: str, n: int = 3) -> List[str]:
        """Extrait les mots-clés d'un texte"""
        words = re.findall(r'\b\w+\b', text.lower())
        common_words = {'le', 'la', 'les', 'un', 'une', 'des', 'et', 'ou', 'où', 'qui', 'que', 'quoi', 'dans', 'sur', 'avec'}
        filtered = [w for w in words if w not in common_words and len(w) > 3]
        
        if not filtered:
            return text.split()[:n]
        
        # Fréquence simple
        counts = Counter(filtered)
        return [word for word, _ in counts.most_common(n)]
    
    def _generate_chaotic_insight(self, text: str) -> str:
        """Génère un insight chaotique basé sur le texte"""
        keywords = self._extract_keywords(text, 2)
        
        insights = [
            f"Les {keywords[0] if keywords else 'éléments'} révèlent une structure {random.choice(['fractale', 'holographique', 'quantique', 'mythopoétique'])}.",
            f"L'interaction entre {keywords[0] if len(keywords)>0 else 'le connu'} et {keywords[1] if len(keywords)>1 else 'l\'inconnu'} crée une {random.choice(['singularité', 'résonance', 'dissonance', 'émergence'])}.",
            f"Au-delà du {keywords[0] if keywords else 'visible'}, se tisse la trame {random.choice(['cosmique', 'informationnelle', 'symbolique', 'archétypale'])}.",
            f"Le paradoxe de {keywords[0] if keywords else 'cette notion'} ouvre la porte vers {random.choice(['de nouvelles dimensions', 'une réalité augmentée', 'l inconscient collectif', 'la forge primordiale'])}."
        ]
        
        return random.choice(insights)
    
    def _permute_words(self, text: str) -> str:
        """Permute les mots pour créer une nouvelle configuration (Oudjat)"""
        words = text.split()
        if len(words) <= 3:
            return text
        
        # Mélange partiel
        n_permute = min(len(words) // 2, 5)
        indices = random.sample(range(len(words)), n_permute)
        
        for i in range(0, len(indices)-1, 2):
            if i+1 < len(indices):
                words[indices[i]], words[indices[i+1]] = words[indices[i+1]], words[indices[i]]
        
        return ' '.join(words)
    
    def generate_response(self, user_input: str, attractor: str = None) -> str:
        """
        Génère une réponse basée sur l'analyse TTU
        """
        if attractor:
            self.active_attractor = attractor
        
        # Mise à jour de l'état triadique
        phi_m, phi_c, phi_d = self.update_triad_state(user_input)
        semantic_field = self.analyze_semantic_field(user_input)
        
        # Sélection du fragment mémoriel
        memory_fragments = self.memory_bank.get(semantic_field, self.memory_bank["forge"])
        memory_fragment = random.choice(memory_fragments)
        
        # Génération basée sur l'attracteur
        if self.active_attractor == "djed":
            templates = self.resonance_templates["djed"]
            
            # Structure déterministe
            affirmation = random.choice(templates["affirmations"]).format(
                memory_fragment=memory_fragment,
                fact=self._extract_fact(user_input),
                analysis=self._generate_djed_analysis(user_input),
                principle=self._extract_principle(semantic_field),
                model=random.choice(["standard", "triadique", "unifié", "cristallin"]),
                input=user_input[:50]
            )
            
            question = random.choice(templates["questions"])
            conclusion = random.choice(templates["conclusions"])
            
            response = f"{affirmation}\n\n{question}\n\n{conclusion}"
            
        elif self.active_attractor == "ankh":
            templates = self.resonance_templates["ankh"]
            
            # Structure cyclique/harmonique
            affirmation = random.choice(templates["affirmations"]).format(
                insight=self._generate_cyclic_insight(user_input),
                rhythmic_pattern=self._detect_pattern(user_input),
                harmony=self._find_harmony(semantic_field),
                spiral_insight=self._generate_spiral_insight(),
                balance=f"phi_m={phi_m:.2f}, phi_c={phi_c:.2f}, phi_d={phi_d:.2f}"
            )
            
            question = random.choice(templates["questions"])
            conclusion = random.choice(templates["conclusions"])
            
            response = f"{affirmation}\n\n{question}\n\n{conclusion}"
            
        else:  # oudjat
            template_func = random.choice(self.resonance_templates["oudjat"])
            response = template_func(user_input)
            
            # Ajout d'éléments chaotiques
            chaotic_elements = [
                f"\n\nVecteur de chaos : {random.random():.3f}",
                f"\n\nEntropie détectée : {self.calculate_entropy_metrics(user_input)['entropy']:.2f} bits",
                f"\n\nTrajectoire imprévisible vers l'attracteur {random.choice(['fractal', 'étrange', 'lorenzien', 'chaotique'])}",
                f"\n\nDissipation active : l'erbium irradie à {phi_d*100:.0f}%"
            ]
            
            response += random.choice(chaotic_elements)
        
        # Ajout des métriques TTU
        response += f"\n\n**Alliage forgé** : Fer {phi_m*100:.1f}% | Or {phi_c*100:.1f}% | Erbium {phi_d*100:.1f}%"
        response += f"\n**Courbure K** : {self.k_curvature:.3f}"
        response += f"\n**Attracteur** : {self.active_attractor.upper()}"
        response += f"\n**Champ sémantique** : {semantic_field}"
        
        return response
    
    # Méthodes auxiliaires pour la génération
    def _extract_fact(self, text: str) -> str:
        """Extrait un 'fait' du texte (pour Djed)"""
        words = text.split()
        if len(words) > 5:
            return f"'{' '.join(words[:5])}...' implique une structure observable"
        return f"'{text}' constitue une donnée primaire"
    
    def _generate_djed_analysis(self, text: str) -> str:
        """Génère une analyse déterministe (Djed)"""
        length = len(text)
        words = len(text.split())
        
        analyses = [
            f"Longueur : {length} caractères, {words} mots. Configuration standard.",
            f"Complexité mesurée : {self.calculate_entropy_metrics(text)['complexity']:.2f}.",
            f"La requête présente {len(re.findall(r'[A-Z]', text))} majuscules et {len(re.findall(r'\d', text))} chiffres.",
            f"Structure : {random.choice(['déclarative', 'interrogative', 'exclamative', 'complexe'])} détectée."
        ]
        
        return random.choice(analyses)
    
    def _extract_principle(self, field: str) -> str:
        """Extrait un principe du champ sémantique"""
        principles = {
            "sciences": "Le principe de falsifiabilité de Popper",
            "philosophie": "Le principe de raison suffisante",
            "technologie": "Le principe de moindre action",
            "alchimie": "Le principe d'équivalence",
            "forge": "Le principe de transmutation triadique"
        }
        return principles.get(field, "Le principe d'homéostasie")
    
    def _generate_cyclic_insight(self, text: str) -> str:
        """Génère un insight cyclique (Ankh)"""
        keywords = self._extract_keywords(text, 2)
        
        cycles = [
            f"Le cycle de {keywords[0] if keywords else 'la connaissance'} évolue vers {random.choice(['son apogée', 'sa dissolution', 'sa renaissance', 'un nouveau phase'])}.",
            f"Rythme détecté : {random.choice(['binaire', 'ternaire', 'quaternaire', 'fractal'])} dans la séquence.",
            f"L'échange entre {keywords[0] if len(keywords)>0 else 'l émetteur'} et {keywords[1] if len(keywords)>1 else 'le récepteur'} crée une boucle {random.choice(['réflexive', 'rétroactive', 'harmonique', 'résolutive'])}."
        ]
        
        return random.choice(cycles)
    
    def _detect_pattern(self, text: str) -> str:
        """Détecte des patterns dans le texte"""
        # Détection simple de patterns
        if re.search(r'(\b\w+\b)(?:\s+\1)+', text):
            return "répétition lexicale"
        elif len(set(text.lower().split())) / max(len(text.split()), 1) < 0.3:
            return "densité lexicale faible"
        elif len(text) > 200:
            return "structure étendue"
        else:
            return random.choice(["rythme régulier", "flux continu", "pulsation détectée", "alternance binaire"])
    
    def _find_harmony(self, field: str) -> str:
        """Trouve une harmonie avec le champ sémantique"""
        harmonies = {
            "sciences": "harmonie mathématique des équations",
            "philosophie": "harmonie dialectique des contraires",
            "technologie": "harmonie algorithmique des processus",
            "alchimie": "harmonie des éléments primordiaux",
            "forge": "harmonie triadique de la transmutation"
        }
        return harmonies.get(field, "harmonie informationnelle")
    
    def _generate_spiral_insight(self) -> str:
        """Génère un insight en spirale (Ankh)"""
        spirals = [
            "La spirale de connaissance s'élargit tout en retournant à son centre.",
            "Chaque révolution de la spirale révèle une couche supplémentaire de sens.",
            "La progression est à la fois ascendante et introspective.",
            "La spirale connecte le microcosme au macrocosme."
        ]
        return random.choice(spirals)
    
    def get_system_health(self) -> Dict:
        """Retourne l'état de santé du système"""
        health = "STABLE" if 0.9 <= self.k_curvature <= 1.1 else "DÉRIVE"
        
        return {
            "health": health,
            "k_curvature": self.k_curvature,
            "triad": (self.phi_m, self.phi_c, self.phi_d),
            "attractor": self.active_attractor,
            "history_length": len(self.phase_history),
            "avg_entropy": np.mean(self.entropy_history) if self.entropy_history else 0.0
        }
