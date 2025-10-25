"""
Générateur de Données Cardiovasculaires - Sénégal
Basé sur l'enquête STEPS 2020-2024 et les études épidémiologiques sénégalaises
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Configuration des seeds pour reproductibilité
np.random.seed(42)
random.seed(42)

class CardioDataGeneratorSenegal:
    """Générateur de données cardiovasculaires réalistes pour le Sénégal"""
    
    def __init__(self, n_patients=1000000):
        self.n_patients = n_patients
        
        # Régions du Sénégal
        self.regions = ['Dakar', 'Thiès', 'Saint-Louis', 'Diourbel', 'Louga', 
                       'Fatick', 'Kaolack', 'Kolda', 'Matam', 'Tambacounda',
                       'Ziguinchor', 'Sédhiou', 'Kaffrine', 'Kédougou']
        
        # Prévalences basées sur enquête STEPS Sénégal
        self.prevalences = {
            'hypertension': 0.282,      # 28.2%
            'diabete': 0.042,           # 4.2%
            'cholesterol_eleve': 0.266, # 26.6%
            'tabagisme': 0.06,          # 6%
            'sedentarite': 0.861,       # 86.1%
            'obesite': 0.11,            # 11%
            'consommation_sel': 0.758,  # 75.8%
            'avc': 0.161,               # 16.1% historique
            'infarctus': 0.036          # 3.6%
        }
        
        # Professions (secteur informel dominant au Sénégal)
        self.professions = {
            'Secteur informel': 0.60,
            'Fonctionnaire': 0.10,
            'Commerce': 0.15,
            'Agriculture': 0.08,
            'Pêche': 0.04,
            'Sans emploi': 0.03
        }
        
    def generate_demographics(self):
        """Génère les données démographiques"""
        print(" Génération des données démographiques...")
        
        data = {}
        
        # ID Patient
        data['patient_id'] = [f'SN{str(i).zfill(6)}' for i in range(1, self.n_patients + 1)]
        
        # Sexe (69% femmes selon études)
        data['sexe'] = np.random.choice(['F', 'M'], self.n_patients, p=[0.69, 0.31])
        
        # Âge (distribution réaliste 18-80 ans)
        ages = []
        for _ in range(self.n_patients):
            if random.random() < 0.3:  # 30% jeunes (18-35)
                age = np.random.randint(18, 36)
            elif random.random() < 0.5:  # 35% adultes (36-55)
                age = np.random.randint(36, 56)
            else:  # 35% seniors (56-80)
                age = np.random.randint(56, 81)
            ages.append(age)
        data['age'] = ages
        
        # Région
        region_weights = [0.25] + [0.75 / 13] * 13  # Dakar = 25%, autres = reste
        data['region'] = np.random.choice(self.regions, self.n_patients, p=region_weights)
        
        # Milieu (urbain/rural)
        data['milieu'] = ['Urbain' if r in ['Dakar', 'Thiès', 'Saint-Louis'] 
                         else 'Semi-urbain' if random.random() < 0.3 
                         else 'Rural' 
                         for r in data['region']]
        
        # Profession
        professions_list = list(self.professions.keys())
        professions_probs = list(self.professions.values())
        data['profession'] = np.random.choice(professions_list, self.n_patients, 
                                             p=professions_probs)
        
        # Niveau d'éducation
        education_levels = ['Aucun', 'Primaire', 'Secondaire', 'Supérieur']
        education_probs = [0.35, 0.30, 0.25, 0.10]
        data['niveau_education'] = np.random.choice(education_levels, self.n_patients,
                                                    p=education_probs)
        
        return data
    
    def generate_clinical_data(self, demographics):
        """Génère les données cliniques"""
        print(" Génération des données cliniques...")
        
        data = demographics.copy()
        ages = np.array(data['age'])
        sexes = np.array(data['sexe'])
        
        # Pression artérielle (avec corrélation à l'âge)
        # Systolique
        pas_base = 115 + (ages - 18) * 0.5 + np.random.normal(0, 15, self.n_patients)
        pas_base = np.clip(pas_base, 90, 200)
        
        # Diastolique
        pad_base = 75 + (ages - 18) * 0.3 + np.random.normal(0, 10, self.n_patients)
        pad_base = np.clip(pad_base, 60, 130)
        
        data['pression_arterielle_systolique'] = np.round(pas_base, 0).astype(int)
        data['pression_arterielle_diastolique'] = np.round(pad_base, 0).astype(int)
        
        # Hypertension (PAS >= 140 ou PAD >= 90)
        data['hypertension'] = ((pas_base >= 140) | (pad_base >= 90)).astype(int)
        
        # IMC (Indice de Masse Corporelle)
        imc_base = 22 + (ages - 18) * 0.08 + np.random.normal(0, 4, self.n_patients)
        # Femmes ont tendance à avoir IMC plus élevé au Sénégal
        imc_base = np.where(sexes == 'F', imc_base + 1.5, imc_base)
        imc_base = np.clip(imc_base, 15, 45)
        data['imc'] = np.round(imc_base, 1)
        
        # Obésité (IMC >= 30)
        data['obesite'] = (imc_base >= 30).astype(int)
        
        # Tour de taille (obésité abdominale)
        tour_taille_base = 75 + (imc_base - 22) * 2 + np.random.normal(0, 8, self.n_patients)
        tour_taille_base = np.where(sexes == 'F', tour_taille_base + 5, tour_taille_base)
        data['tour_taille_cm'] = np.round(np.clip(tour_taille_base, 60, 140), 0).astype(int)
        
        # Glycémie à jeun (mg/dL)
        glycemie_base = 85 + (ages - 18) * 0.3 + np.random.normal(0, 15, self.n_patients)
        glycemie_base = np.clip(glycemie_base, 60, 250)
        data['glycemie_jeun_mg_dl'] = np.round(glycemie_base, 0).astype(int)
        
        # Diabète (glycémie >= 126 mg/dL)
        data['diabete'] = (glycemie_base >= 126).astype(int)
        
        # Cholestérol total (mg/dL)
        cholesterol_base = 170 + (ages - 18) * 0.4 + np.random.normal(0, 30, self.n_patients)
        data['cholesterol_total_mg_dl'] = np.round(np.clip(cholesterol_base, 120, 320), 0).astype(int)
        
        # Cholestérol élevé (>= 200 mg/dL)
        data['cholesterol_eleve'] = (cholesterol_base >= 200).astype(int)
        
        # HDL (bon cholestérol)
        hdl_base = 50 + np.random.normal(0, 12, self.n_patients)
        data['hdl_mg_dl'] = np.round(np.clip(hdl_base, 25, 80), 0).astype(int)
        
        # LDL (mauvais cholestérol)
        ldl = data['cholesterol_total_mg_dl'] - data['hdl_mg_dl'] - 30
        data['ldl_mg_dl'] = np.clip(ldl, 50, 200).astype(int)
        
        # Triglycérides
        trigly_base = 120 + (imc_base - 22) * 4 + np.random.normal(0, 40, self.n_patients)
        data['triglycerides_mg_dl'] = np.round(np.clip(trigly_base, 50, 400), 0).astype(int)
        
        # Fréquence cardiaque au repos
        fc_base = 70 + np.random.normal(0, 10, self.n_patients)
        data['frequence_cardiaque_repos'] = np.round(np.clip(fc_base, 50, 110), 0).astype(int)
        
        return data
    
    def generate_lifestyle_data(self, clinical_data):
        """Génère les données de style de vie"""
        print("🏃 Génération des données de style de vie...")
        
        data = clinical_data.copy()
        
        # Tabagisme (6% au Sénégal, majoritairement hommes)
        sexes = np.array(data['sexe'])
        tabagisme_prob = np.where(sexes == 'M', 0.15, 0.01)
        data['tabagisme'] = np.random.binomial(1, tabagisme_prob)
        
        # Cigarettes par jour (pour fumeurs)
        data['cigarettes_par_jour'] = np.where(
            data['tabagisme'] == 1,
            np.random.randint(1, 20, self.n_patients),
            0
        )
        
        # Consommation d'alcool (3.4%)
        data['consommation_alcool'] = np.random.binomial(1, 0.034, self.n_patients)
        
        # Activité physique (86.1% insuffisante)
        data['activite_physique_suffisante'] = np.random.binomial(1, 0.139, self.n_patients)
        
        # Minutes d'activité physique par semaine
        data['minutes_activite_semaine'] = np.where(
            data['activite_physique_suffisante'] == 1,
            np.random.randint(150, 400, self.n_patients),
            np.random.randint(0, 100, self.n_patients)
        )
        
        # Sédentarité (heures assises par jour)
        sedentarite_base = np.random.uniform(2, 12, self.n_patients)
        # Plus de sédentarité en milieu urbain
        milieux = np.array(data['milieu'])
        sedentarite_base = np.where(milieux == 'Urbain', sedentarite_base + 2, sedentarite_base)
        data['heures_sedentaire_jour'] = np.round(np.clip(sedentarite_base, 1, 16), 1)
        
        # Consommation de sel et bouillons (75.8%)
        data['sel_bouillon_excessif'] = np.random.binomial(1, 0.758, self.n_patients)
        
        # Consommation de fruits et légumes (portions/jour)
        data['portions_fruits_legumes_jour'] = np.round(
            np.random.gamma(2, 1.5, self.n_patients), 1
        )
        
        # Consommation de sucre (morceaux au petit déjeuner)
        sucre_prob = np.random.rand(self.n_patients)
        data['morceaux_sucre_matin'] = np.where(
            sucre_prob < 0.291,  # 29.1% consomment >= 3 morceaux
            np.random.randint(3, 7, self.n_patients),
            np.random.randint(0, 3, self.n_patients)
        )
        
        return data
    
    def generate_medical_history(self, lifestyle_data):
        """Génère l'historique médical et événements cardiovasculaires"""
        print(" Génération de l'historique médical...")
        
        data = lifestyle_data.copy()
        ages = np.array(data['age'])
        
        # Calculer score de risque cardiovasculaire
        risk_score = (
            data['hypertension'] * 2 +
            data['diabete'] * 2 +
            data['cholesterol_eleve'] * 1.5 +
            data['tabagisme'] * 1.5 +
            data['obesite'] * 1 +
            (1 - data['activite_physique_suffisante']) * 1 +
            (ages > 55).astype(int) * 2
        ) / 10
        
        # AVC (Accident Vasculaire Cérébral) - plus probable avec âge et risques
        avc_prob = np.clip(risk_score * 0.05, 0, 0.25)
        data['antecedent_avc'] = np.random.binomial(1, avc_prob)
        
        # Infarctus du myocarde
        infarctus_prob = np.clip(risk_score * 0.03, 0, 0.15)
        data['antecedent_infarctus'] = np.random.binomial(1, infarctus_prob)
        
        # Insuffisance cardiaque
        ic_prob = np.clip((risk_score * 0.04) * (ages > 50).astype(int), 0, 0.20)
        data['insuffisance_cardiaque'] = np.random.binomial(1, ic_prob)
        
        # Maladie rénale chronique (4.3%)
        mrc_prob = 0.043 * (1 + risk_score * 0.5)
        data['maladie_renale_chronique'] = np.random.binomial(1, np.clip(mrc_prob, 0, 0.15))
        
        # Antécédents familiaux cardiovasculaires
        data['antecedents_familiaux_cardio'] = np.random.binomial(1, 0.25, self.n_patients)
        
        # Douleur thoracique
        douleur_prob = risk_score * 0.15
        data['douleur_thoracique'] = np.random.binomial(1, np.clip(douleur_prob, 0, 0.4))
        
        # Type de douleur thoracique
        types_douleur = ['Aucune', 'Angine typique', 'Angine atypique', 'Douleur non-angineuse']
        data['type_douleur_thoracique'] = np.where(
            data['douleur_thoracique'] == 1,
            np.random.choice(types_douleur[1:], self.n_patients),
            'Aucune'
        )
        
        # Dyspnée (essoufflement)
        dyspnee_prob = risk_score * 0.20
        data['dyspnee'] = np.random.binomial(1, np.clip(dyspnee_prob, 0, 0.5))
        
        # Traitement en cours
        data['traitement_antihypertenseur'] = np.where(
            data['hypertension'] == 1,
            np.random.binomial(1, 0.65, self.n_patients),
            0
        )
        
        data['traitement_diabete'] = np.where(
            data['diabete'] == 1,
            np.random.binomial(1, 0.70, self.n_patients),
            0
        )
        
        data['traitement_cholesterol'] = np.where(
            data['cholesterol_eleve'] == 1,
            np.random.binomial(1, 0.45, self.n_patients),
            0
        )
        
        # Score de risque cardiovasculaire (0-10)
        data['score_risque_cardiovasculaire'] = np.round(np.clip(risk_score * 10, 0, 10), 1)
        
        # Risque cardiovasculaire catégoriel
        data['categorie_risque'] = pd.cut(
            data['score_risque_cardiovasculaire'],
            bins=[-0.1, 2, 4, 6, 10],
            labels=['Faible', 'Modéré', 'Élevé', 'Très élevé']
        )
        
        # Événement cardiovasculaire (cible prédictive)
        event_prob = np.clip(risk_score * 0.25, 0, 0.6)
        data['evenement_cardiovasculaire'] = np.random.binomial(1, event_prob)
        
        return data
    
    def generate_temporal_data(self, medical_data):
        """Ajoute les données temporelles"""
        print(" Ajout des données temporelles...")
        
        data = medical_data.copy()
        
        # Date de consultation (entre 2020 et 2024)
        start_date = datetime(2020, 1, 1)
        end_date = datetime(2024, 12, 31)
        date_range = (end_date - start_date).days
        
        dates = [start_date + timedelta(days=random.randint(0, date_range)) 
                for _ in range(self.n_patients)]
        
        data['date_consultation'] = dates
        data['annee_consultation'] = [d.year for d in dates]
        data['mois_consultation'] = [d.month for d in dates]
        
        # Saison (impact sur certaines pathologies)
        data['saison'] = pd.cut(
            data['mois_consultation'],
            bins=[0, 3, 6, 9, 12],
            labels=['Saison sèche froide', 'Saison sèche chaude', 'Hivernage', 'Post-hivernage']
        )
        
        return data
    
    def generate_complete_dataset(self):
        """Génère le dataset complet"""
        print("\n" + "="*60)
        print("🇸🇳 GÉNÉRATION DU DATASET CARDIOVASCULAIRE SÉNÉGAL")
        print("="*60 + "\n")
        
        # Étape 1: Démographie
        demographics = self.generate_demographics()
        
        # Étape 2: Données cliniques
        clinical = self.generate_clinical_data(demographics)
        
        # Étape 3: Style de vie
        lifestyle = self.generate_lifestyle_data(clinical)
        
        # Étape 4: Historique médical
        medical = self.generate_medical_history(lifestyle)
        
        # Étape 5: Données temporelles
        complete = self.generate_temporal_data(medical)
        
        # Créer DataFrame
        df = pd.DataFrame(complete)
        
        print("\n Dataset généré avec succès!")
        print(f" Nombre total de patients: {len(df)}")
        print(f" Nombre de variables: {len(df.columns)}")
        
        return df
    
    def display_statistics(self, df):
        """Affiche les statistiques du dataset"""
        print("\n" + "="*60)
        print(" STATISTIQUES DU DATASET")
        print("="*60 + "\n")
        
        print(" DÉMOGRAPHIE:")
        print(f"   • Femmes: {(df['sexe'] == 'F').sum()} ({(df['sexe'] == 'F').mean()*100:.1f}%)")
        print(f"   • Hommes: {(df['sexe'] == 'M').sum()} ({(df['sexe'] == 'M').mean()*100:.1f}%)")
        print(f"   • Âge moyen: {df['age'].mean():.1f} ans (± {df['age'].std():.1f})")
        print(f"   • Région Dakar: {(df['region'] == 'Dakar').mean()*100:.1f}%")
        
        print("\n PRÉVALENCES DES MALADIES:")
        print(f"   • Hypertension: {df['hypertension'].mean()*100:.1f}%")
        print(f"   • Diabète: {df['diabete'].mean()*100:.1f}%")
        print(f"   • Cholestérol élevé: {df['cholesterol_eleve'].mean()*100:.1f}%")
        print(f"   • Obésité: {df['obesite'].mean()*100:.1f}%")
        print(f"   • AVC (antécédent): {df['antecedent_avc'].mean()*100:.1f}%")
        print(f"   • Infarctus (antécédent): {df['antecedent_infarctus'].mean()*100:.1f}%")
        
        print("\n FACTEURS DE RISQUE:")
        print(f"   • Tabagisme: {df['tabagisme'].mean()*100:.1f}%")
        print(f"   • Consommation d'alcool: {df['consommation_alcool'].mean()*100:.1f}%")
        print(f"   • Activité physique insuffisante: {(1-df['activite_physique_suffisante']).mean()*100:.1f}%")
        print(f"   • Consommation excessive sel/bouillon: {df['sel_bouillon_excessif'].mean()*100:.1f}%")
        
        print("\n RISQUE CARDIOVASCULAIRE:")
        print(df['categorie_risque'].value_counts())
        print(f"\n   • Événements cardiovasculaires: {df['evenement_cardiovasculaire'].mean()*100:.1f}%")
        
        print("\n" + "="*60)


# === EXÉCUTION ===
if __name__ == "__main__":
    # Générer le dataset
    generator = CardioDataGeneratorSenegal(n_patients=1000000)
    df_cardio_senegal = generator.generate_complete_dataset()
    
    # Afficher les statistiques
    generator.display_statistics(df_cardio_senegal)
    
    # Aperçu du dataset
    print("\n APERÇU DES DONNÉES:")
    print(df_cardio_senegal.head(10))
    
    print("\n INFORMATIONS DU DATASET:")
    print(df_cardio_senegal.info())
    
    # Sauvegarder le dataset
    output_file = 'dataset_cardiovasculaire_senegal.csv'
    df_cardio_senegal.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n Dataset sauvegardé: {output_file}")
    print(f" Taille du fichier: {len(df_cardio_senegal)} lignes × {len(df_cardio_senegal.columns)} colonnes")
    
    print("\n TERMINÉ!")