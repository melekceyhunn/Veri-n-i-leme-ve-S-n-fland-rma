import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
def label_encoder(df):
    label_encoders = {}
    
    for column in df.select_dtypes(include='object').columns:
        label_encoder = LabelEncoder()
        df[column] = label_encoder.fit_transform(df[column].astype(str))
        label_encoders[column] = label_encoder
    
    return df, label_encoders
def preprocessing(data):
    df = pd.read_excel(data)
    print("EKSİK VERİ DOLDURMA")
    eksik_veri = df.columns[df.isnull().any()]
    for column in eksik_veri:
        if df[column].dtype == 'object':
            most_frequent_value = df[column].mode()[0]
            df[column] = df[column].fillna(most_frequent_value)
        else:
            ort_deger = df[column].mean()
            df[column] = df[column].fillna(ort_deger)

    print("Dataframe son hali:",df)
    print("AYKIRI VERİLERİ ORTALAMA İLE GİDERME")
    sayisal_stunlar = df.select_dtypes(include='number').columns
    for column in sayisal_stunlar:
        df[column] = pd.to_numeric(df[column], errors='coerce')
    for column in sayisal_stunlar:
        plt.figure(figsize=(8, 6))
        sns.boxplot(x=df[column])
        plt.title(f'Box Plot - {column}')
        plt.show()
    #IQR'ı hesaplama işlemi
    Q1 = df[sayisal_stunlar].quantile(0.25)
    Q3 = df[sayisal_stunlar].quantile(0.75)
    IQR = Q3 - Q1
    aykiri_degerler = (df[sayisal_stunlar] < (Q1 - 1.5 * IQR)) | (df[sayisal_stunlar] > (Q3 + 1.5 * IQR))
    aykiriDeger_sayac = aykiri_degerler.sum().sum()
    print("Toplam Aykırı Değer Sayısı:", aykiriDeger_sayac)
    aykiri_veriler = df[aykiri_degerler.any(axis=1)]
    print("Aykırı Veriler (IQR):")
    print(aykiri_veriler)
    #Ortalama ile aykırı verileri doldurma işlemi
    aykirisiz_df = df.copy()
    for column in sayisal_stunlar:
        ort_deger = aykirisiz_df[column].mean()
        aykirisiz_df.loc[aykiri_degerler[column], column] = ort_deger
    print("\nAykırı Değerleri Ortalama ile Doldurma Sonucu:")
    print(aykirisiz_df)
    print("İKİLİLEŞTİRME")
    feature1 = input("İlk sayısal özelliği girin (verisetindeki ismini '' olmadan girin): ")
    feature2 = input("İkinci sayısal özelliği girin (verisetindeki ismini '' olmadan girin): ")
    plt.figure(figsize=(8, 6))
    plt.scatter(aykirisiz_df[feature1], aykirisiz_df[feature2])
    plt.title(f'{feature1} ve {feature2} İlişkisi')
    plt.xlabel(feature1)
    plt.ylabel(feature2)
    plt.grid(True)
    plt.show()
    print("DÜZLEŞTİRİLMİŞ VERİ")
    duzlestirilmis_df = pd.melt(df, var_name='Özellik', value_name='Değer')
    print(duzlestirilmis_df)
    
    print("KATEGORİK VERİLERİN LABELENCODER İLE KODLANMASI")
    df_encoded, label_encoders = label_encoder(aykirisiz_df)
    print(df_encoded)
    print(label_encoders)
    return df_encoded
def test(data_path, model_path="C:\\Users\\mlkce\\OneDrive\\Masaüstü\\veri_bilimi_final_projesi\\classification.pkl"):
    data = preprocessing(data_path)
    x_test = data.drop('Fiyat', axis=1)
    y_test = data['Fiyat']
    
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)

    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    
    return accuracy, report
def tanimlayici_veri_analizi(data_path):
    print("**********************VERİSETİNİN TANIMLAYICI VERİ ANALİZİNİN YAPILMASI***************************")
    df = pd.read_excel(data_path)
    kayit_sayisi =( len(df)+1)
    print(f"Kayıt Sayısı: {kayit_sayisi}\n")

    nitelik_sayisi = len(df.columns)
    print(f"Nitelik Sayısı: {nitelik_sayisi}")
    print("Nitelik Türleri:")
    print(df.dtypes, '\n')

    print("Merkezi Eğilim Ölçüleri:")
    print(df.describe().loc[['mean', '50%']], '\n')

    print("Merkezden Dağılım Ölçüleri:")
    print(df.describe().loc[['std', 'min', '25%', '50%', '75%', 'max']], '\n')

    print("Beş Sayı Özeti:")
    print(df.describe().loc[['min', '25%', '50%', '75%', 'max']])
def veri_gorsellestirme_analiz(data_path):
    print("********************VERİNİN GÖRSELLEŞTİRİLME İLE ANALİZİNİN YAPILMASI***************************")
    df = pd.read_excel(data_path)
    ozellikler = df.columns
    print("Kullanılabilir Özellikler:")
    for i, ozellik in enumerate(ozellikler):
        print(f"{i + 1}. {ozellik}")
    gorsellestirilecek_index = int(input("Görselleştirmek istediğiniz özelliği seçin (1,2,...): ")) - 1
    gorsellestirilecek = ozellikler[gorsellestirilecek_index]

    ilk_secilen_ozellik_index = int(input("Analizinin yapılmasını istediğiniz ilk özelliği seçin (1,2,...): ")) - 1
    ilk_secilen_ozellik = ozellikler[ilk_secilen_ozellik_index]

    ikinci_secilen_ozellik_index = int(input("Analizinin yapılmasını istediğiniz ikinci özelliği seçin (1,2,...): ")) - 1
    ikinci_secilen_ozellik = ozellikler[ikinci_secilen_ozellik_index]
    #GÖRSELLEŞTİRME
    if pd.api.types.is_numeric_dtype(df[gorsellestirilecek]):
        plt.figure(figsize=(10, 6))
        sns.histplot(df[gorsellestirilecek], bins=20, kde=True)
        plt.title(f'{gorsellestirilecek} Histogramı')
        plt.xlabel(gorsellestirilecek)
        plt.ylabel('Frekans')
        plt.show()
    else:
        plt.figure(figsize=(10, 6))
        sns.countplot(x=gorsellestirilecek, data=df)
        plt.title(f'{gorsellestirilecek} Dağılımı')
        plt.xlabel(gorsellestirilecek)
        plt.ylabel('Sayı')
        plt.show()

    # ANALİZ YAPMA
    if pd.api.types.is_numeric_dtype(df[ilk_secilen_ozellik]) and pd.api.types.is_numeric_dtype(df[ikinci_secilen_ozellik]):
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=ilk_secilen_ozellik, y=ikinci_secilen_ozellik, data=df)
        plt.title(f'{ilk_secilen_ozellik} ve {ikinci_secilen_ozellik} İlişkisi')
        plt.xlabel(ilk_secilen_ozellik)
        plt.ylabel(ikinci_secilen_ozellik)
        plt.show()
    else:
        plt.figure(figsize=(12, 8))
        sns.boxplot(x=ilk_secilen_ozellik, y=ikinci_secilen_ozellik, data=df)
        plt.title(f'{ilk_secilen_ozellik} ve {ikinci_secilen_ozellik} İlişkisi')
        plt.xlabel(ilk_secilen_ozellik)
        plt.ylabel(ikinci_secilen_ozellik)
        plt.xticks(rotation=45)
        plt.show()
data_path='file/path'
tanimlayici_veri_analizi(data_path)
veri_gorsellestirme_analiz(data_path)
accuracy, report = test(data_path)
print(f"Accuracy: {accuracy}")
print("Classification Report:\n", report)    