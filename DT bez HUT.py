import pandas as pd
from datetime import datetime
import re
import numpy as np
import warnings
import matplotlib.pyplot as plt
from collections import Counter
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
file_path = r"C:\Users\admin\PycharmProjects\pythonProject3\databaza.xlsx"
df = pd.read_excel(file_path, engine="openpyxl")
df_transposed = df.T
df_transposed.columns = df_transposed.iloc[0]
df_transposed = df_transposed[1:]
columns_to_drop = [
    "ECHO", "ECHOKG", "EMG", "EEG", "EKG", "EKG Holter", "Holter", "USG ECC",
    "CT mozgu", "MRI mozgu", "DG", "*DG:", "Záver ()"
]
df_transposed = df_transposed.drop(columns=[col for col in columns_to_drop if col in df_transposed.columns], errors='ignore')
df_transposed = df_transposed[~df_transposed.apply(lambda row: row.astype(str).str.contains("Poznámky", na=False)).any(axis=1)]
if "Navigácia" in df_transposed.columns:
    df_transposed.drop(columns=["Navigácia"], inplace=True)
df_transposed.reset_index(drop=True, inplace=True)
date_columns = [
    "Dátum",
    "Začiatok ťažkostí (približný dátum)?",
    "Kedy bolo posledné odpadnutie?",
    "Kedy boli ťažkosti najhoršie?"
]
for col in date_columns:
    if col in df_transposed.columns:
        df_transposed[col] = pd.to_datetime(df_transposed[col], errors="coerce").dt.date
if "Dátum" in df_transposed.columns:
    df_transposed["Dátum"] = pd.to_datetime(df_transposed["Dátum"], errors='coerce').dt.date  # Преобразуем в формат даты (без времени)
if df_transposed["Dátum"].dtype == "object":
    ref_year = min(df_transposed["Dátum"].dropna()).year
else:
    ref_year = 2000

def fix_invalid_dates(df):
    """Исправляет некорректные даты и заменяет NaN на среднее значение (не меньше 2017 года)."""
    df["Dátum"] = pd.to_datetime(df["Dátum"], errors='coerce').dt.date
    valid_dates = df["Dátum"].dropna()
    if valid_dates.empty:
        print("Ошибка: В колонке 'Dátum' нет корректных значений для вычисления среднего.")
        return df
    mean_timestamp = pd.to_datetime(valid_dates).mean()
    mean_date = mean_timestamp.date()
    df["Dátum"] = df["Dátum"].apply(lambda x: mean_date if pd.isna(x) or x.year < 2017 else x)
    return df
df_transposed = fix_invalid_dates(df_transposed)

def extract_birthdate_and_gender(rodne_cislo):
    if pd.isna(rodne_cislo):
        return np.nan, np.nan

    rodne_cislo = str(rodne_cislo).replace("/", "").strip()  # Удаляем '/', лишние пробелы

    if not re.fullmatch(r'\d{6,10}', rodne_cislo):  # Проверяем формат
        return np.nan, np.nan

    try:
        year = int(rodne_cislo[:2])
        month = int(rodne_cislo[2:4])
        day = int(rodne_cislo[4:6])
        if len(rodne_cislo) == 10:
            serial_number = int(rodne_cislo[6:9])
            if serial_number < 500:
                year += 1900
            else:
                year += 2000
        else:
            year += 1900 if year >= 20 else 2000
        if month > 50:
            gender = "F"
            month -= 50
        else:
            gender = "M"
        birth_date = datetime(year, month, day).date()
        return birth_date, gender
    except ValueError:
        return np.nan, np.nan

if "Rodné číslo" in df_transposed.columns:
    df_transposed["Datum narodenia"], df_transposed["Pohlavie"] = zip(
        *df_transposed["Rodné číslo"].apply(lambda x: extract_birthdate_and_gender(str(x))))

df_transposed["Vek"] = df_transposed.apply(
    lambda row: (row["Dátum"].year - row["Datum narodenia"].year) if pd.notna(row["Dátum"]) and pd.notna(row["Datum narodenia"]) else np.nan,
    axis=1
)
df_transposed["Vek"] = df_transposed["Vek"].apply(lambda x: -1 if pd.isna(x) or x < 0 else x)

start_col = "Aké lieky užívate"
end_col = "Datum narodenia"

if start_col in df_transposed.columns and end_col in df_transposed.columns:
    start_idx = df_transposed.columns.get_loc(start_col)
    end_idx = df_transposed.columns.get_loc(end_col)
    lieky_columns = df_transposed.iloc[:, start_idx:end_idx]
    df_transposed["Aké lieky užívate"] = lieky_columns.apply(
        lambda row: ', '.join(row.dropna().astype(str).replace(["-1", "nan", "Názov"], "").str.strip()).strip(", "),
        axis=1
    )
    columns_to_drop = df_transposed.columns[start_idx+1:end_idx]
    df_transposed.drop(columns=columns_to_drop, inplace=True)
df_transposed["Aké lieky užívate"] = df_transposed["Aké lieky užívate"].replace("", "0")

df_transposed["Synkopa"] = df_transposed["Záver HUT"].apply(lambda x: 1 if isinstance(x, str) and re.search(r"VASIS ", x, re.IGNORECASE) else 0)

def get_syncope_type(text):
    if not isinstance(text, str):
        return -1
    text = text.upper()
    type_mapping = {
        r"VASIS I|1\b": "VASIS I",
        r"VASIS (IIA|2A)\b": "VASIS IIa",
        r"VASIS (IIB|2B|ILB|IIB)\b": "VASIS IIb",
        r"VASIS (III|3)\b": "VASIS III",
    }
    for pattern, syncope_type in type_mapping.items():
        if re.search(pattern, text):
            return syncope_type
    return -1

df_transposed["Typ Synkopy"] = df_transposed.apply(lambda row: get_syncope_type(row["Záver HUT"]) if row["Synkopa"] == 1 else "NO CLASS", axis=1)
df_transposed = df_transposed.drop(columns=["Záver HUT"], errors='ignore')
hut_test_column = "HUT Test"

datum_index = df_transposed.columns.get_loc("Dátum")
hut_test_index = df_transposed.columns.get_loc(hut_test_column)

columns_to_move = ["Datum narodenia", "Pohlavie", "Vek", "Synkopa", "Typ Synkopy"]
existing_columns_to_move = [col for col in columns_to_move if col in df_transposed.columns]

new_column_order = (
    df_transposed.columns[:datum_index + 1].tolist() +
    existing_columns_to_move +
    [col for col in df_transposed.columns[datum_index + 1:] if col not in existing_columns_to_move]
)

df_transposed = df_transposed[new_column_order]
df_transposed = df_transposed.drop(columns=["Rodné číslo"], errors='ignore')
def clean_text(value):
    if isinstance(value, str):
        value = value.strip().lower()
        value = value.replace("nienie", "nie")
        value = value.replace("áno", "ano")
        value = value.replace("niw", "nie")
        value = value.strip().upper()
    return value
df_transposed = df_transposed.map(clean_text)
translation_rules = {
    'ANO': 1,
    'NIE': 0,
    'X': 1,
    'NEUVEDENE': -1,
    'POZIT': 1,
    'NEGAT': 0,
    'POZIT.':1,
    'NEGAT.': 0,
    'NEUVEDENÉ': -1,
    '': -1
}
df_transposed = df_transposed.replace(translation_rules)
def replace_non_numeric_with_one(df: pd.DataFrame, column_name: str) -> None:
    df[column_name] = df[column_name].apply(lambda x: 1 if not str(x).replace('.', '', 1).isdigit() else x)
# Применяем к колонке 'Ine' в df_transposed
replace_non_numeric_with_one(df_transposed, 'Iné')
df_transposed.fillna(-1, inplace=True)
def convert_floats_to_int_inplace(df: pd.DataFrame) -> None:

    for col in df.columns:
        if pd.api.types.is_float_dtype(df[col]):
            df[col] = df[col].astype('Int64')

convert_floats_to_int_inplace(df_transposed)
column_counts = Counter(df_transposed.columns)
new_columns = []
seen = {}
for col in df_transposed.columns:
    if column_counts[col] > 1:
        if col in seen:
            seen[col] += 1
        else:
            seen[col] = 1
        new_col_name = f"{col}_{seen[col]}"
    else:
        new_col_name = col
    new_columns.append(new_col_name)
df_transposed.columns = new_columns
df_transposed.rename(columns={
    'HUT Test': 'A',
    'Vykonaný áno/nie': 'A1',
    'Vstupný tlak': 'A2',
    'Vstupný pulz': 'A3',
    'Tlak po sklopení': 'A4',
    'Pulz po sklopení': 'A5',
    'Tlak po podaní NTG': 'A6',
    'Pulz po podaní NTG': 'A7',
    'Tlak po ukončení (po sklopení)': 'A8',
    'Pulz po ukončení (po sklopení)': 'A9',
    'Výsledok Testu': 'A10',
    'Som vyšetrovaný pre': 'B',
    'Stratu vedomia': 'B1',
    'Pocity hroziacej straty vedomia': 'B2',
    'Stav po resuscitácii': 'B3',
    'Stav po eplileptickom záchvate': 'B4',
    'Opakované pády': 'B5',
    'Moje ťažkosti': 'C',
    'Začiatok ťažkostí (približný dátum)?': 'C1',
    'Počet odpadnutí spolu?': 'C2',
    'Kedy bolo posledné odpadnutie?': 'C3',
    'Kedy boli ťažkosti najhoršie?': 'C4',
    'Popíšte ťažkosti / V akej situácii vznikli': 'D',
    'Strata vedomia pri státí?': 'D1',
    'Strata vedomia do 1 minúty po postavení sa?': 'D2',
    'Pri chôdzi?': 'D3',
    'Pri fyzickej námahe (akej?)': 'D4',
    'Strata vedomia v sede?': 'D5',
    'Strata vedomia poležačky?': 'D6',
    'Čo viedlo k strate vedomia': 'E',
    'Preľudnené priestory?': 'E1',
    'Dusné prostredie?': 'E2',
    'Teplé prostredie?': 'E3',
    'Pohľad na krv?': 'E4',
    'Nepríjemné emócie (strach, úzkosť, rozrušenie, odpor, pohľad na násilie)': 'E5',
    'Medicínsky výkon?': 'E6',
    'Bolesť?': 'E7',
    'Dehydratácia?': 'E8',
    'Menštruácia?': 'E9',
    'Strata krvi?': 'E10',
    'Vznikla strata vedomia pri niektorej z týchto situácií?': 'F',
    'Pri stolici': 'F1',
    'Pri močení': 'F2',
    'Pri kašli': 'F3',
    'Pri kýchaní/smrkaní nosa': 'F4',
    'Pri jedení/prehĺtaní': 'F5',
    'Po náhlej bolesti': 'F6',
    'Počas fyzickej námahy': 'F7',
    'Pri hlade': 'F8',
    'Pri nedostatku spánku, únave': 'F9',
    'Iné1': 'F10',
    'Užili ste hodinu pred stratou vedomia nejaké lieky alebo alkohol?': 'G',
    'Čo ste cítili tesne pred stratou vedomia?': 'H',
    'Pocit na zvracanie alebo zvracanie': 'H1',
    'Pocit tepla/horúco': 'H2',
    'Pot': 'H3',
    'Zahmlievanie pred očami': 'H4',
    'Hučanie v ušiach': 'H5',
    'Búšenie srdca1': 'H6',
    'Búšenie srdca_2': 'H7',
    'Bolesť na hrudníku': 'H8',
    'Neobvyklý zápach': 'H9',
    'Neobvyklé zvuky': 'H10',
    'Poruchy reči alebo slabosť polovice tela': 'H11',
    'Nepociťoval som nič zvláštne': 'H12',
    'Nepamätám sa1': 'H13',
    'Iné2': 'H14',
    'Ako dlho trvali tieto pocity pred stratou vedomia?': 'I',
    'Niekoľko sekúnd1': 'I1',
    'Do 1 miníty': 'I2',
    'Do 5 minút1': 'I3',
    'Viac ako 5 minút1': 'I4',
    'Čo ste urobili pri hroziacej strate vedomia?': 'J',
    'Sadol som si': 'J1',
    'Ľahol som si': 'J2',
    'Nestihol som urobiť nič pretože som stratil vedomie': 'J3',
    'Ak boli prítomní svedkovia, ako dlho podľa nich trvalo bezvedomie?': 'K',
    'Niekoľko sekúnd2': 'K1',
    'Do minúty': 'K2',
    'Do 5 minút2': 'K3',
    'Viac ako 5 minút2': 'K4',
    'Mali ste kŕče počas bezvedomia (v prípade svedkov udalosti)?': 'L',
    'Odišla vám stolica alebo moč počas bezvedomia?': 'M',
    'Pamätáte si na udalosti po strate vedomia?': 'N',
    'Mali ste pohryzený jazyk, pery?': 'N1',
    'Udreli ste sa pri páde, boli ste poranení v dôsledku pádu?': 'N2',
    'Po prebratí ste podľa údajov svedkov boli viac ako 30 minút dozerientovaní?': 'N3',
    'Bolela vás hlava alebo svaly?': 'N4',
    'Aj po prebratí ste pociťovali nevoľnosť?': 'N5',
    'Cítili ste sa normálne': 'N6',
    'Nepamätáte sa2': 'N7',
    'Výskyt ochorení vo vašej rodine': 'O',
    'Náhle úmrtie člena rodiny (v akom veku?)': 'O1',
    'Ochorenie srdca_1': 'O2',
    'Ochorenie srdca_2': 'O3',
    'Srdcová arytmia/kardiostimulátor': 'O4',
    'Ochoria mozgu/epilepsia': 'O5',
    'Na aké ochorenia ste sa doteraz liečili?': 'P',
    'Ochorenie srdca_3': 'P1',
    'Ochorenie srdca_4': 'P2',
    'Ochorenie chlopní': 'P3',
    'Srdcová slabosť': 'P4',
    'Koronárna chorova srdca': 'P5',
    'Srdcové arytmie': 'P6',
    'Búšenie srdca2': 'P7',
    'Búšenie srdca_4': 'P8',
    'Bolesti na hrudníku': 'P9',
    'Vysoký tlak krvi': 'P10',
    'Nízky tlak krvi': 'P11',
    'Závraty': 'P12',
    'Ochorenia obličiek': 'P13',
    'Diabetes (cukrovka)': 'P14',
    'Anémia': 'P15',
    'Astma': 'P16',
    'Ochorenia pľúc': 'P17',
    'Ochorenia priedušiek': 'P18',
    'Ochorenia žalúdka': 'P19',
    'Ochorenia čreva': 'P20',
    'Ochorenia štítnej žľazy': 'P21',
    'Endokrinologické ochorenia': 'P22',
    'Bolesti hlavy': 'P23',
    'Neurologické ochorenia': 'P24',
    'Parkinsonová choroba': 'P25',
    'Psychiatrické ochorenia ': 'P26',
    'Depresia': 'P27',
    'Ochorenia krčnej chrbtice': 'P28',
    'Bolesti chrbta': 'P29',
    'Reumatologické ochorenia': 'P30',
    'Nádorové ochorenie': 'P31',
    'Prekonané operácie': 'P32',
    'Prekonané úrazy': 'P33',
    'Alergie': 'P34',
    'Aké vyšetrenia ste doteraz absolvovali kvôli stratám vedomia?': 'Q',
    'Kardiologické vyšetrenia': 'Q1',
    'Záťažový test (bicyklová ergometria)': 'Q2',
    'Koronografické vyšetrenie': 'Q3',
    'HUT test': 'Q4',
    'Pažerákovú stimuláciu': 'Q5',
    'Invazívne vyšetrenie arytmií (EFV)': 'Q6',
    'Nukleárne vyšetrenie srdca (SPECT)': 'Q7',
    'CT srdca': 'Q8',
    'MRI srdca': 'Q9',
    'Neurologické vyšetrenia': 'Q10',
    'USG mozgových ciev': 'Q11',
    'CT alebo MRI mozgu': 'Q12',
    'RTG, CT alebo MRI krčnej chrbtice': 'Q13',
    'Elektromyografia (EMG)': 'Q14',
    'Psychiatrické vyšetrenie': 'Q15',
    'Endokrinologické vyšetrenie': 'Q16',
    'Odber krvi': 'Q17',
    'Iné3': 'Q18',
    'Boli ste v poslednom období očkovaní (cca za posledných 10-15 rokov)?': 'R',
    'Proti HPV (rakovina krčka maternice)': 'R1',
    'Proti chrípke': 'R2',
    'Iné':'R3',
    'Aké lieky užívate': 'S',
}, inplace=True)
def process_blocks_final_fixed_v2(df):
    df_transformed = df.copy()
    block_columns = [col for col in df.columns if re.match(r'^[A-Z]\d+$', col)]

    blocks = {}
    for col in block_columns:
        block = re.match(r'^([A-Z])', col).group(1)
        blocks.setdefault(block, []).append(col)
    if 'A' in blocks:
        blocks['A'] = [col for col in blocks['A'] if col not in ['A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9']]
        if not blocks['A']:
            blocks.pop('A')
    for block, cols in blocks.items():
        if block in df.columns and len(cols) > 0:
            subset = df_transformed[cols].copy()

            has_one = subset.eq(1).any(axis=1)

            all_negative_one = subset.eq(-1).all(axis=1)

            # Применяем изменения
            df_transformed.loc[has_one, cols] = subset.loc[has_one].where(subset.loc[has_one] == 1, 0)
            df_transformed.loc[has_one, block] = 1

            df_transformed.loc[all_negative_one, cols] = -1
            df_transformed.loc[all_negative_one, block] = -1

    return df_transformed

df_transformed = process_blocks_final_fixed_v2(df_transposed)
def calculate_and_replace_age(df, birthdate_col, event_cols):

    for col in event_cols:
        if col in df.columns:
            df[col] = df.apply(
                lambda row: calculate_age(row[birthdate_col], row[col]), axis=1
            )
    return df

def calculate_age(birthdate, event_date):
    try:
        birthdate = str(birthdate)
        birth_year = int(birthdate[:4])
        birth_month = int(birthdate[5:7])
        birth_day = int(birthdate[8:10])
        event_date = str(event_date)
        event_year = int(event_date[:4])
        event_month = int(event_date[5:7])
        event_day = int(event_date[8:10])
        age = event_year - birth_year
        if (event_month, event_day) < (birth_month, birth_day):
            age -= 1
        return max(age, -1)
    except (ValueError, TypeError):
        return -1
event_date_columns = ["C1", "C3", "C4"]
df_transformed = calculate_and_replace_age(df_transformed, "Datum narodenia", event_date_columns)
def calculate_mean_age_difference_single(df, event_col, age_col):
    valid_rows = df[(df[event_col] != -1) & (df[age_col] != -1)]
    if not valid_rows.empty:
        return (valid_rows[age_col] - valid_rows[event_col]).mean()
    return 0
mean_age_difference_C1 = calculate_mean_age_difference_single(df_transformed, "C1", "Vek")
mean_age_difference_C3 = calculate_mean_age_difference_single(df_transformed, "C3", "Vek")
mean_age_difference_C4 = calculate_mean_age_difference_single(df_transformed, "C4", "Vek")

df_transformed["C1"] = df_transformed.apply(
    lambda row: mean_age_difference_C1 if row["C1"] == -1 else row["C1"],
    axis=1
)
df_transformed["C3"] = df_transformed.apply(
    lambda row: row["Vek"] - mean_age_difference_C3 if row["C3"] == -1 else row["C3"],
    axis=1
)
df_transformed["C4"] = df_transformed.apply(
    lambda row: row["Vek"] - mean_age_difference_C4 if row["C4"] == -1 else row["C4"],
    axis=1
)
df_transformed["C1"] = df_transformed["C1"].round().astype(float)
df_transformed["C3"] = df_transformed["C3"].round().astype(float)
df_transformed["C4"] = df_transformed["C4"].round().astype(float)
df_transformed["Vek"] = df_transformed["Vek"].round().astype(float)
df_transformed["P1"] = df_transformed["P1"].round().astype(int)
df_transformed["P2"] = df_transformed["P2"].round().astype(int)
df_transformed["O3"] = df_transformed["O3"].round().astype(int)
df_transformed["O2"] = df_transformed["O2"].round().astype(int)
df_transformed.drop(columns=['A10','A9','O2', 'O3', 'O4', 'O5','F9', 'F6', 'F1', 'F2', 'F3', 'F4', 'F5', 'F8', 'F7', 'F10','K3', 'K2', 'K1', 'Dátum', 'Datum narodenia','S','Číslo dotazníka','A','B','C','D','E','F','G','H','I','J','K','N','O','P','Q','R'], inplace=True)
df_transformed.to_csv(r"C:\Users\admin\anaconda\envs\new_env\data_full.csv", index=False)
df = df_transformed
df_full = df_transformed.copy()
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import shap
import matplotlib.pyplot as plt

# --- Preprocessing ---
if 'Pohlavie' in df.columns:
    df['Pohlavie'] = df['Pohlavie'].map({'M': 0, 'F': 1})

blood_pressure_cols = ['A2', 'A4', 'A6', 'A8']
pulse_cols = ['A3', 'A5', 'A7', 'A9']

for col in blood_pressure_cols:
    if col in df.columns:
        df[[f'{col}_systolic', f'{col}_diastolic']] = df[col].astype(str).str.extract(r'(\d+)/(\d+)').astype(float)
        df.drop(columns=[col], inplace=True)

for col in pulse_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

block_A_cols = [
    'A1', 'A9',
    'A2_systolic', 'A2_diastolic', 'A3',
    'A4_systolic', 'A4_diastolic', 'A5',
    'A6_systolic', 'A6_diastolic', 'A7',
    'A8_systolic', 'A8_diastolic'
]
df.drop(columns=[col for col in block_A_cols if col in df.columns], inplace=True)

# --- Príprava datasetu ---
df = df.select_dtypes(include=[float, int])
df_clean = df[df["Synkopa"].isin([0, 1])].copy()
df_clean["Synkopa"] = df_clean["Synkopa"].astype(int)

X = df_clean.drop(columns=["Synkopa", "Typ Synkopy"], errors='ignore').copy()
y = df_clean["Synkopa"]
X = X.fillna(-1).astype(float)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# --- Decision Tree s GridSearch ---
param_grid_dt = {
    'max_depth': [5, 10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}

grid_dt = GridSearchCV(
    estimator=DecisionTreeClassifier(random_state=42, class_weight='balanced'),
    param_grid=param_grid_dt,
    cv=StratifiedKFold(3),
    scoring='f1_weighted',
    n_jobs=-1
)
grid_dt.fit(X_train, y_train)
best_dt = grid_dt.best_estimator_
from sklearn.tree import export_text, plot_tree
import matplotlib.pyplot as plt

# --- Список бинарных и числовых признаков ---
binary_cols = [col for col in X.columns if set(X[col].dropna().unique()).issubset({0, 1, -1})]
numeric_cols = [col for col in X.columns if col not in binary_cols]

# --- Функция логичного отображения правил ---
def logical_export_text(decision_tree, feature_names, binary_cols):
    rules_text = export_text(decision_tree, feature_names=feature_names)
    lines = rules_text.split('\n')
    new_lines = []

    for line in lines:
        if "<=" in line or ">" in line:
            for col in binary_cols:
                if f"{col} <= 0.50" in line or f"{col} <= 0.5" in line:
                    line = line.replace(f"{col} <= 0.50", f"{col} = 0").replace(f"{col} <= 0.5", f"{col} = 0")
                elif f"{col} >  0.50" in line or f"{col} >  0.5" in line:
                    line = line.replace(f"{col} >  0.50", f"{col} = 1").replace(f"{col} >  0.5", f"{col} = 1")
        new_lines.append(line)
    return '\n'.join(new_lines)

# --- Оценка качества модели ---
y_pred = best_dt.predict(X_test)
y_proba = best_dt.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted')
roc_auc = roc_auc_score(y_test, y_proba)

print(f"Decision Tree - Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1 Score: {f1:.3f}, AUC: {roc_auc:.3f}")

# --- Визуализация дерева ---
plt.figure(figsize=(50, 40))
plot_tree(
    best_dt,
    feature_names=X.columns,
    class_names=["Negatívny", "Pozitívny"],
    filled=True,
    rounded=True,
    fontsize=10,
    max_depth=None
)
plt.title("Vizualizácia rozhodovacieho stromu pre predikciu synkopy (HUT test)")
plt.tight_layout()
plt.savefig("strom.png", dpi=300)

from sklearn.tree import _tree

def get_rules(tree, feature_names, binary_cols, target_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    paths = []
    path = []

    def recurse(node, path, paths):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]

            if name in binary_cols:
                # Для бинарных признаков (0 или 1)
                recurse(tree_.children_left[node],
                        path + [f"({name} = 0)"], paths)
                recurse(tree_.children_right[node],
                        path + [f"({name} = 1)"], paths)
            else:
                # Для числовых признаков
                recurse(tree_.children_left[node],
                        path + [f"({name} <= {threshold:.2f})"], paths)
                recurse(tree_.children_right[node],
                        path + [f"({name} > {threshold:.2f})"], paths)
        else:
            # Лист дерева (решение)
            value = tree_.value[node][0]
            class_idx = value.argmax()
            probability = value[class_idx] / value.sum()
            path_statement = " AND ".join(path)
            rule = f"IF {path_statement} THEN class = {target_names[class_idx]} (probability = {probability:.2f})"
            paths.append(rule)

    recurse(0, path, paths)

    return paths

# Список бинарных признаков (только 0/1)
binary_cols = [col for col in X.columns if set(X[col].dropna().unique()).issubset({0, 1, -1})]

# Получение логичных правил:
rules = get_rules(best_dt, feature_names=X.columns, binary_cols=binary_cols, target_names=["Negatívny", "Pozitívny"])

# Печать первых 20 правил (для примера)
for i, rule in enumerate(rules[:100], start=1):
    print(f"R{i}: {rule}")
# --- Post-pruning pomocou cost-complexity pruning ---
from sklearn.tree import DecisionTreeClassifier
import numpy as np

# Построение пути обрезки
path = best_dt.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

# Строим деревья для каждого значения ccp_alpha
dt_models = []
for ccp_alpha in ccp_alphas:
    clf = DecisionTreeClassifier(
        random_state=42,
        class_weight='balanced',
        ccp_alpha=ccp_alpha
    )
    clf.fit(X_train, y_train)
    dt_models.append(clf)

# Вычисляем метрики для всех деревьев
from sklearn.metrics import f1_score

f1_scores = [f1_score(y_test, clf.predict(X_test), average='weighted') for clf in dt_models]

# Находим наилучшее дерево по F1
best_idx = np.argmax(f1_scores)
best_pruned_dt = dt_models[best_idx]

print(f"🎯 Best ccp_alpha: {ccp_alphas[best_idx]:.5f}, F1 Score: {f1_scores[best_idx]:.3f}")

# Сохранить визуализацию дерева в PNG
plt.figure(figsize=(50, 40))
plot_tree(
    best_pruned_dt,
    feature_names=X.columns,
    class_names=["Negatívny", "Pozitívny"],
    filled=True,
    rounded=True,
    fontsize=10
)
plt.title("Orezané rozhodovacie strom (post-pruning s CCP)")
plt.tight_layout()
plt.savefig("strom.png", dpi=300)


# --- Извлечение правил из обрезанного дерева ---
rules_pruned = get_rules(
    best_pruned_dt,
    feature_names=X.columns,
    binary_cols=binary_cols,
    target_names=["Negatívny", "Pozitívny"]
)

# Печать первых 100 правил после pruning
for i, rule in enumerate(rules_pruned[:100], start=1):
    print(f"PRUNED R{i}: {rule}")
# from sklearn.linear_model import LogisticRegression
# # Обучаем модель Logistic Regression
# log_reg = LogisticRegression(max_iter=1000, random_state=42)
# log_reg.fit(X_train, y_train)
# # --- Оценка качества модели ---
# y_pred = log_reg.predict(X_test)
# y_proba = log_reg.predict_proba(X_test)[:, 1]
#
# accuracy = accuracy_score(y_test, y_pred)
# precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
# recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
# f1 = f1_score(y_test, y_pred, average='weighted')
# roc_auc = roc_auc_score(y_test, y_proba)
#
# print(f"Log Reg - Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1 Score: {f1:.3f}, AUC: {roc_auc:.3f}")
# # Извлекаем коэффициенты (важность признаков)
# coefficients = log_reg.coef_[0]  # Получаем коэффициенты для признаков
# selected_features = X.columns  # Признаки, которые использовались для обучения
#
# # Создаём DataFrame для важности признаков
# df_importance = pd.DataFrame({
#     "Feature": selected_features,
#     "Importance": coefficients
# }).sort_values("Importance", ascending=False)
#
# # Визуализация важности признаков
# plt.figure(figsize=(40, 40))
# plt.barh(df_importance["Feature"], df_importance["Importance"], color='purple')
# plt.xlabel("Importances")
# plt.title("Importance Logistic Regression")
# plt.gca().invert_yaxis()  # Чтобы самые важные признаки были сверху
# plt.grid(axis='x', linestyle='--', alpha=0.5)
# plt.tight_layout()
# plt.show()
import joblib
joblib.dump(best_dt, "best_decision_tree_model.pkl")
