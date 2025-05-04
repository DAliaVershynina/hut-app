import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Predikcia pozitivneho HUT testu", layout="wide")
st.title("Predikcia pozitivneho HUT testu pomocou rozhodovacieho stromu")
st.markdown("Vyplňte formulár nižšie a kliknite na **Predikovať**, aby ste zistili pravdepodobnosť pozitivneho HUT testu.")

@st.cache_resource
def load_model():
    return joblib.load("best_decision_tree_model.pkl")

try:
    model = load_model()
    model_loaded = True
except Exception as e:
    st.error(f"❌ Nepodarilo sa načítať model: {e}")
    model_loaded = False


# Словарь для отображения читаемых названий
labels = {
    'B2': 'Bol/bola vyšetrovaný pre pocity hroziacej straty vedomia(B2)',
    'B3': 'Bol/bola vyšetrovaný pre stav po resuscitácii(B3)',
    'B4': 'Bol/bola vyšetrovaný pre stav po epileptickom záchvate',
    'B5': 'Bol/bola vyšetrovaný pre opakované pády',
    'C3': 'Vek pri poslednom odpadnutí',
    'C4': 'Vek pri najhoršom stave',
    'C1': 'Vek pri začiatku ťažkostí ',
    'D1': 'Tažkosti / V akej situácii vznikli: Strata vedomia pri státí',
    'D2': 'Tažkosti / V akej situácii vznikli: Strata vedomia do 1 minúty po postavení sa',
    'D3': 'Tažkosti / V akej situácii vznikli: Strata vedomia pri chôdzi',
    'D4': 'Tažkosti / V akej situácii vznikli: Strata vedomia pri fyzickej námahe',
    'D5': 'Tažkosti / V akej situácii vznikli: Strata vedomia v sede',
    'D6': 'Tažkosti / V akej situácii vznikli: Strata vedomia poležiačky',
    'E1': 'Čo viedlo k strate vedomia: Preľudnené priestory', 'E2': 'Čo viedlo k strate vedomia: Dusné prostredie', 'E3': 'Čo viedlo k strate vedomia: Teplé prostredie',
    'E4': 'Čo viedlo k strate vedomia: Pohľad na krv', 'E5': 'Čo viedlo k strate vedomia: Nepríjemné emócie', 'E6': 'Čo viedlo k strate vedomia: Medicínsky výkon',
    'E7': 'Čo viedlo k strate vedomia: Bolesť', 'E8': 'Čo viedlo k strate vedomia: Dehydratácia', 'E9': 'Čo viedlo k strate vedomia: Menštruácia', 'E10': 'Čo viedlo k strate vedomia: Strata krvi',
    'H1': 'Cítili tesne pred stratou vedomia: Pocit na zvracanie', 'H2': 'Cítili tesne pred stratou vedomia: Pocit tepla', 'H3': 'Cítili tesne pred stratou vedomia: Potenie',
    'H4': 'Cítili tesne pred stratou vedomia: Zahmlievanie pred očami', 'H5': 'Cítili tesne pred stratou vedomia: Hučanie v ušiach', 'H6': 'Cítili tesne pred stratou vedomia: Búšenie srdca',
    'H8': 'Cítili tesne pred stratou vedomia: Bolesť na hrudi', 'H9': 'Cítili tesne pred stratou vedomia: Neobvyklý zápach', 'H10': 'Cítili tesne pred stratou vedomia: Neobvyklé zvuky',
    'H11': 'Cítili tesne pred stratou vedomia: Poruchy reči / slabosť tela', 'H12': 'Cítili tesne pred stratou vedomia: Nepociťoval som nič zvláštne', 'H13': 'Cítili tesne pred stratou vedomia: Nepamätám sa',
    'I1': 'Ako dlho trvali tieto pocity pred stratou vedomia: Niekoľko sekúnd (pred stratou vedomia)', 'I2': 'Ako dlho trvali tieto pocity pred stratou vedomia: Do 1 minúty',
    'I3': 'Ako dlho trvali tieto pocity pred stratou vedomia: Do 5 minút', 'I4': 'Ako dlho trvali tieto pocity pred stratou vedomia: Viac ako 5 minút',
    'J1': 'Čo ste urobili pri hroziacej strate vedomia: Sadol som si', 'J2': 'Čo ste urobili pri hroziacej strate vedomia: Ľahol som si', 'K4': 'Ak boli prítomní svedkovia, ako dlho podľa nich trvalo bezvedomie: Bezvedomie trvalo viac ako 5 minút',
    'L': 'Kŕče počas bezvedomia', 'M': 'Odišla stolica alebo moč počas bezvedomia?',
    'N2': 'Pamätáte si na udalosti po strate vedomia: Poranenie pri páde', 'N3': 'Pamätáte si na udalosti po strate vedomia: Dezorientácia > 30 minút',
    'N5': 'Pamätáte si na udalosti po strate vedomia: Nevoľnosť po prebratí', 'N6': 'Pamätáte si na udalosti po strate vedomia: Cítil/a sa normálne', 'N7': 'Pamätáte si na udalosti po strate vedomia: Nepamätám sa',
    'P1': 'Na aké ochorenia ste sa doteraz liečili: Ochorenie srdca_3',
    'P2': 'Na aké ochorenia ste sa doteraz liečili: úzkostný stav',
    'P3': 'Na aké ochorenia ste sa doteraz liečili: Ochorenie chlopní',
    'P4': 'Na aké ochorenia ste sa doteraz liečili: Srdcová slabosť',
    'P5': 'Na aké ochorenia ste sa doteraz liečili: Koronárna chorova srdca',
    'P6': 'Na aké ochorenia ste sa doteraz liečili: Srdcové arytmie',
    'P7': 'Na aké ochorenia ste sa doteraz liečili: Búšenie srdca',
    'P9': 'Na aké ochorenia ste sa doteraz liečili: Bolesti na hrudníku',
    'P10': 'Na aké ochorenia ste sa doteraz liečili: Vysoký tlak krvi',
    'P11': 'Na aké ochorenia ste sa doteraz liečili: Nízky tlak krvi',
    'P12': 'Na aké ochorenia ste sa doteraz liečili: Závraty',
    'P13': 'Na aké ochorenia ste sa doteraz liečili: Ochorenia obličiek',
    'P14': 'Na aké ochorenia ste sa doteraz liečili: Diabetes (cukrovka)',
    'P15': 'Na aké ochorenia ste sa doteraz liečili: Anémia',
    'P16': 'Na aké ochorenia ste sa doteraz liečili: Astma',
    'P17': 'Na aké ochorenia ste sa doteraz liečili: Ochorenia pľúc',
    'P18': 'Na aké ochorenia ste sa doteraz liečili: Ochorenia priedušiek',
    'P19': 'Na aké ochorenia ste sa doteraz liečili: Ochorenia žalúdka',
    'P20': 'Na aké ochorenia ste sa doteraz liečili: Ochorenia čreva',
    'P21': 'Na aké ochorenia ste sa doteraz liečili: Ochorenia štítnej žľazy',
    'P22': 'Na aké ochorenia ste sa doteraz liečili: Endokrinologické ochorenia',
    'P23': 'Na aké ochorenia ste sa doteraz liečili: Bolesti hlavy',
    'P24': 'Na aké ochorenia ste sa doteraz liečili: Neurologické ochorenia',
    'P25': 'Na aké ochorenia ste sa doteraz liečili: Parkinsonová choroba',
    'P26': 'Na aké ochorenia ste sa doteraz liečili: Psychiatrické ochorenia ',
    'P27': 'Na aké ochorenia ste sa doteraz liečili: Depresia',
    'P28': 'Na aké ochorenia ste sa doteraz liečili: Ochorenia krčnej chrbtice',
    'P29': 'Na aké ochorenia ste sa doteraz liečili: Bolesti chrbta',
    'P30': 'Na aké ochorenia ste sa doteraz liečili: Reumatologické ochorenia',
    'P31': 'Na aké ochorenia ste sa doteraz liečili: Nádorové ochorenie',
    'P32': 'Na aké ochorenia ste sa doteraz liečili: Prekonané operácie',
    'P33': 'Na aké ochorenia ste sa doteraz liečili: Prekonané úrazy',
    'P34': 'Na aké ochorenia ste sa doteraz liečili: Alergie',
    'Q1': 'Aké vyšetrenia ste doteraz absolvovali kvôli stratám vedomia: Kardiologické vyšetrenia',
    'Q2': 'Aké vyšetrenia ste doteraz absolvovali kvôli stratám vedomia: Záťažový test (bicyklová ergometria)',
    'Q3': 'Aké vyšetrenia ste doteraz absolvovali kvôli stratám vedomia: Koronografické vyšetrenie',
    'Q4': 'Aké vyšetrenia ste doteraz absolvovali kvôli stratám vedomia: HUT test',
    'Q5': 'Aké vyšetrenia ste doteraz absolvovali kvôli stratám vedomia: Pažerákovú stimuláciu',
    'Q6': 'Aké vyšetrenia ste doteraz absolvovali kvôli stratám vedomia: Invazívne vyšetrenie arytmií (EFV)',
    'Q7': 'Aké vyšetrenia ste doteraz absolvovali kvôli stratám vedomia: Nukleárne vyšetrenie srdca (SPECT)',
    'Q8': 'Aké vyšetrenia ste doteraz absolvovali kvôli stratám vedomia: CT srdca',
    'Q9': 'Aké vyšetrenia ste doteraz absolvovali kvôli stratám vedomia: MRI srdca',
    'Q10': 'Aké vyšetrenia ste doteraz absolvovali kvôli stratám vedomia: Neurologické vyšetrenia',
    'Q11': 'Aké vyšetrenia ste doteraz absolvovali kvôli stratám vedomia: USG mozgových ciev',
    'Q12': 'Aké vyšetrenia ste doteraz absolvovali kvôli stratám vedomia: CT alebo MRI mozgu',
    'Q13': 'Aké vyšetrenia ste doteraz absolvovali kvôli stratám vedomia: RTG, CT alebo MRI krčnej chrbtice',
    'Q14': 'Aké vyšetrenia ste doteraz absolvovali kvôli stratám vedomia: Elektromyografia (EMG)',
    'Q15': 'Aké vyšetrenia ste doteraz absolvovali kvôli stratám vedomia: Psychiatrické vyšetrenie',
    'Q16': 'Aké vyšetrenia ste doteraz absolvovali kvôli stratám vedomia: Endokrinologické vyšetrenie',
    'Q17': 'Aké vyšetrenia ste doteraz absolvovali kvôli stratám vedomia: Odber krvi',
    'Q18': 'Aké vyšetrenia ste doteraz absolvovali kvôli stratám vedomia: Iné',
    'R1': 'Boli ste v poslednom období očkovaní (cca za posledných 10-15 rokov): Proti HPV (rakovina krčka maternice)',
    'R2': 'Boli ste v poslednom období očkovaní (cca za posledných 10-15 rokov): Proti chrípke',
    'R3': 'Boli ste v poslednom období očkovaní (cca za posledných 10-15 rokov): Iné'
}
if model_loaded:
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            vek = st.number_input("Vek pacienta", min_value=0, max_value=120, value=35)
            pohlavie = st.selectbox("Pohlavie", options=["Muž", "Žena"])
            c1 = st.number_input("Vek pri začiatku ťažkostí (C1)", min_value=0, max_value=120, value=30)
            c3 = st.number_input("Vek pri poslednom odpadnutí (C3)", min_value=0, max_value=120, value=32)
            c4 = st.number_input("Vek pri najhoršom stave (C4)", min_value=0, max_value=120, value=33)

        binary_features = [
            'B2','B3','B4','B5','D1','D2','D3','D4','D5','D6','E1','E2','E3','E4','E5','E6','E7','E8','E9','E10',
            'H1','H2','H3','H4','H5','H6','H8','H9','H10','H11','H12','H13','I1','I2','I3','I4','J1','J2','K4',
            'L','M','N2','N3','N5','N6','N7','P1','P2','P3','P4','P5','P6','P7','P9','P10','P11','P12','P13',
            'P14','P15','P16','P17','P18','P19','P20','P21','P22','P23','P24','P25','P26','P27','P28','P29',
            'P30','P31','P32','P33','P34','Q1','Q2','Q3','Q4','Q5','Q6','Q7','Q8','Q9','Q10','Q11','Q12','Q13',
            'Q14','Q15','Q16','Q17','Q18','R1','R2','R3']

        binary_inputs = {}
        with col2:
            for i, feature in enumerate(binary_features):
                label = labels.get(feature, feature)
                binary_inputs[feature] = st.radio(label, ["Nie", "Áno"], horizontal=True, index=0)

        submitted = st.form_submit_button("Predikovať")

        if submitted:
            input_data = {
                'Vek': vek,
                'Pohlavie': 0 if pohlavie == "Muž" else 1,
                'C1': c1,
                'C3': c3,
                'C4': c4,
            }
            for k, v in binary_inputs.items():
                input_data[k] = 1 if v == "Áno" else 0

            X_input = pd.DataFrame([input_data])
            X_input = X_input[model.feature_names_in_]

            pred = model.predict(X_input)[0]
            proba = model.predict_proba(X_input)[0][1]

            if pred == 1:
                st.success(f"Vysoká pravdepodobnosť pozitivneho HUT testu (pravdepodobnosť = {proba:.2f})")
            else:
                st.warning(f"Nízka pravdepodobnosť pozitivneho HUT testu (pravdepodobnosť = {proba:.2f})")

            import shap

            # Инициализация TreeExplainer
            explainer = shap.TreeExplainer(model)

            # 🔁 Подготовка входа к ожидаемому виду
            expected_features = model.feature_names_in_

            # Добавим отсутствующие признаки как -1
            for col in expected_features:
                if col not in X_input.columns:
                    X_input[col] = -1

            # Упорядочим признаки
            X_input = X_input[expected_features]

            import shap
            import numpy as np
            import pandas as pd
            import matplotlib.pyplot as plt

            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(X_input)

            shap_array = np.array(shap_vals)

            # 💡 Поддержка формата (1, 105, 2)
            if shap_array.ndim == 3 and shap_array.shape == (1, 105, 2):
                shap_effect = shap_array[0, :, 1]  # 1 pacient, všetky znaky, SHAP pre class 1
            elif shap_array.ndim == 3 and shap_array.shape[0] == 2:
                shap_effect = shap_array[1, 0, :]  # fallback pre [2, 1, 105]
            elif shap_array.ndim == 2:
                shap_effect = shap_array[0]
            else:
                st.error(f"Neznámy formát SHAP hodnôt: shape={shap_array.shape}")
                st.stop()

            # 💥 Проверка длины
            if len(shap_effect) != len(X_input.columns):
                st.error(
                    f"Chyba: Počet SHAP hodnôt ({len(shap_effect)}) sa nezhoduje s počtom vstupných znakov ({len(X_input.columns)}).")
                st.stop()

            # 📊 Визуализация
            shap_df = pd.DataFrame({
                'Feature': list(X_input.columns),
                'SHAP Value': shap_effect.tolist()
            }).sort_values(by='SHAP Value', key=abs, ascending=False).head(20)

            st.subheader("🔍 Interpretácia rozhodnutia modelu")

            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(10, 6))
            # Заменим коды признаков на понятные подписи
            shap_df["Label"] = shap_df["Feature"].map(labels).fillna(shap_df["Feature"])
            ax.barh(shap_df['Label'], shap_df['SHAP Value'])
            ax.set_xlabel("SHAP hodnota (vplyv na predikciu)")
            ax.set_title("Top 20 najvplyvnejších znakov na rozhodnutie modelu")
            plt.gca().invert_yaxis()
            st.pyplot(fig)
