import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import json
from PIL import Image
import io

# Налаштування сторінки
st.set_page_config(page_title="Fashion MNIST Classifier", layout="wide", page_icon="👗")

# Назви класів Fashion MNIST
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

class_names_ukr = ['Футболка', 'Штани', 'Светр', 'Сукня', 'Пальто',
                   'Сандалі', 'Сорочка', 'Кросівки', 'Сумка', 'Черевики']


# Завантаження моделі та історії
@st.cache_resource
def load_model(model_name):
    """Завантажує модель за назвою"""
    model_paths = {
        'Згорткова нейромережа (CNN)': 'fashion_mnist_cnn.keras',
        'VGG16 Transfer Learning': 'fashion_mnist_vgg16.keras'
    }
    try:
        model = tf.keras.models.load_model(model_paths[model_name])
        return model
    except Exception as e:
        st.error(f"Помилка завантаження моделі: {e}")
        return None


@st.cache_data
def load_history(model_name):
    """Завантажує історію навчання"""
    history_paths = {
        'Згорткова нейромережа (CNN)': 'history_cnn.json',
        'VGG16 Transfer Learning': 'history_vgg16.json'
    }
    try:
        with open(history_paths[model_name], 'r') as f:
            return json.load(f)
    except Exception as e:
        st.warning(f"Історія навчання недоступна: {e}")
        return None


def preprocess_image(image, model_name):
    """Попередня обробка зображення залежно від моделі"""
    # Конвертація в градації сірого
    img = image.convert('L')

    # Зміна розміру та формату залежно від моделі
    if model_name == 'VGG16 Transfer Learning':
        # VGG16 потребує 32x32 RGB
        img = img.resize((32, 32))
        img_array = np.array(img) / 255.0
        # Конвертація в RGB
        img_array = np.stack([img_array] * 3, axis=-1)
        img_array = np.expand_dims(img_array, axis=0)
    else:  # CNN
        # CNN потребує 28x28 grayscale
        img = img.resize((28, 28))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=-1)
        img_array = np.expand_dims(img_array, axis=0)

    return img_array, img


def plot_training_history(history, model_name):
    """Відображає графіки навчання"""
    if history is None:
        st.warning("Історія навчання недоступна")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history['accuracy']) + 1)

    # Графік точності
    ax1.plot(epochs, history['accuracy'], 'b-', label='Точність на тренуванні', linewidth=2.5, marker='o', markersize=4)
    ax1.plot(epochs, history['val_accuracy'], 'g-', label='Точність на валідації', linewidth=2.5, marker='s',
             markersize=4)
    ax1.set_title(f'Точність моделі ({model_name})', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Епохи', fontsize=12)
    ax1.set_ylabel('Точність', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_ylim([0.7, 1.0])

    # Графік втрат
    ax2.plot(epochs, history['loss'], 'r--', label='Втрати на тренуванні', linewidth=2.5, marker='o', markersize=4)
    ax2.plot(epochs, history['val_loss'], 'orange', linestyle='--', label='Втрати на валідації', linewidth=2.5,
             marker='s', markersize=4)
    ax2.set_title(f'Втрати моделі ({model_name})', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Епохи', fontsize=12)
    ax2.set_ylabel('Втрати', fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    return fig


def predict_image(model, img_array):
    """Робить передбачення для зображення"""
    predictions = model.predict(img_array, verbose=0)
    return predictions[0]


# ============ ГОЛОВНА ЧАСТИНА ЗАСТОСУНКУ ============

# Заголовок
st.markdown("""
    <h1 style='text-align: center; color: #2E86AB;'>
        👗 Fashion MNIST Класифікатор
    </h1>
    <h3 style='text-align: center; color: #666;'>
        Порівняння згорткової нейромережі та VGG16
    </h3>
""", unsafe_allow_html=True)

st.markdown("---")

# Бічна панель
st.sidebar.header("⚙️ Вибір моделі")

model_name = st.sidebar.radio(
    "Оберіть модель для класифікації:",
    ['Згорткова нейромережа (CNN)', 'VGG16 Transfer Learning'],
    help="Оберіть модель для аналізу зображень"
)

# Інформація про моделі
model_info = {
    'Згорткова нейромережа (CNN)': {
        'icon': '🧠',
        'description': 'Згорткова мережа з 3 Conv2D шарами, MaxPooling та Dense шарами',
        'architecture': '3 Conv2D → 2 MaxPooling → Flatten → Dense',
        'accuracy': '~91.83%',
        'parameters': '93,322',
        'training_time': '~3 хв 35 сек',
        'input_size': '28×28 (grayscale)',
        'advantages': 'Швидка, компактна, ефективна для простих зображень'
    },
    'VGG16 Transfer Learning': {
        'icon': '🎯',
        'description': 'Transfer Learning на базі попередньо навченої VGG16 з ImageNet',
        'architecture': 'VGG16 (frozen) → GlobalAveragePooling → Dense → BatchNorm → Dropout',
        'accuracy': '~88.0%',
        'parameters': 'VGG16 (14.7M)',
        'training_time': '~1 год 25 хв',
        'input_size': '32×32 (RGB)',
        'advantages': 'Використовує попередньо навчені ваги, потужні ознаки'
    }
}

current_info = model_info[model_name]

st.sidebar.markdown(f"""
### {current_info['icon']} Інформація про модель

**Опис:**  
{current_info['description']}

**Архітектура:**  
`{current_info['architecture']}`

**Метрики:**
- 📊 Точність: `{current_info['accuracy']}`
- 🔢 Параметри: `{current_info['parameters']}`
- ⏱️ Час навчання: `{current_info['training_time']}`
- 📐 Розмір входу: `{current_info['input_size']}`

**Переваги:**  
{current_info['advantages']}
""")

# Завантаження обраної моделі
model = load_model(model_name)

if model is not None:
    # Вкладки
    tab1, tab2, tab3 = st.tabs(["📊 Історія навчання", "🖼️ Класифікація зображень", "🔍 Архітектура моделі"])

    # ============ ВКЛАДКА 1: ІСТОРІЯ НАВЧАННЯ ============
    with tab1:
        st.header(f"📈 Графіки навчання - {model_name}")

        history = load_history(model_name)
        if history:
            # Графіки
            fig = plot_training_history(history, model_name)
            st.pyplot(fig)

            st.markdown("---")

            # Метрики в колонках
            st.subheader("📊 Фінальні метрики")
            col1, col2, col3, col4 = st.columns(4)

            final_train_acc = history['accuracy'][-1]
            final_val_acc = history['val_accuracy'][-1]
            final_train_loss = history['loss'][-1]
            final_val_loss = history['val_loss'][-1]

            with col1:
                st.metric(
                    "Точність (тренування)",
                    f"{final_train_acc:.4f}",
                    delta=f"{final_train_acc - history['accuracy'][0]:.4f}"
                )
            with col2:
                st.metric(
                    "Точність (валідація)",
                    f"{final_val_acc:.4f}",
                    delta=f"{final_val_acc - history['val_accuracy'][0]:.4f}"
                )
            with col3:
                st.metric(
                    "Втрати (тренування)",
                    f"{final_train_loss:.4f}",
                    delta=f"{final_train_loss - history['loss'][0]:.4f}",
                    delta_color="inverse"
                )
            with col4:
                st.metric(
                    "Втрати (валідація)",
                    f"{final_val_loss:.4f}",
                    delta=f"{final_val_loss - history['val_loss'][0]:.4f}",
                    delta_color="inverse"
                )

            # Додаткова інформація
            st.markdown("---")
            col1, col2 = st.columns(2)

            with col1:
                st.info(f"""
                **📌 Кількість епох:** {len(history['accuracy'])}  
                **🎯 Найкраща точність (val):** {max(history['val_accuracy']):.4f}  
                **📉 Найменші втрати (val):** {min(history['val_loss']):.4f}
                """)

            with col2:
                overfitting = final_train_acc - final_val_acc
                if overfitting > 0.05:
                    st.warning(f"""
                    ⚠️ **Можливе перенавчання**  
                    Різниця між тренувальною та валідаційною точністю: {overfitting:.4f}
                    """)
                else:
                    st.success(f"""
                    ✅ **Модель навчена добре**  
                    Різниця між тренувальною та валідаційною точністю: {overfitting:.4f}
                    """)

    # ============ ВКЛАДКА 2: КЛАСИФІКАЦІЯ ============
    with tab2:
        st.header(f"🖼️ Класифікація зображень - {model_name}")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("📤 Завантаження зображення")

            uploaded_file = st.file_uploader(
                "Оберіть зображення одягу",
                type=['png', 'jpg', 'jpeg', 'bmp'],
                help="Підтримувані формати: PNG, JPG, JPEG, BMP"
            )

            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption="Оригінальне зображення", use_container_width=True)

                # Інформація про зображення
                st.info(f"**Розмір зображення:** {image.size[0]}×{image.size[1]} пікселів")

                # Кнопка для класифікації
                if st.button("🔍 Класифікувати зображення", type="primary", use_container_width=True):
                    with st.spinner("🔄 Обробка зображення..."):
                        # Попередня обробка
                        img_array, processed_img = preprocess_image(image, model_name)

                        # Передбачення
                        predictions = predict_image(model, img_array)
                        predicted_class = np.argmax(predictions)
                        confidence = predictions[predicted_class] * 100

                        # Збереження результатів
                        st.session_state['predictions'] = predictions
                        st.session_state['predicted_class'] = predicted_class
                        st.session_state['confidence'] = confidence
                        st.session_state['processed_img'] = processed_img
                        st.session_state['model_used'] = model_name

                        st.success("✅ Класифікація завершена!")

        with col2:
            if 'predictions' in st.session_state:
                st.subheader("📊 Результати класифікації")

                # Відображення обробленого зображення
                input_size = "32×32 RGB" if st.session_state[
                                                'model_used'] == 'VGG16 Transfer Learning' else "28×28 Grayscale"
                st.image(
                    st.session_state['processed_img'],
                    caption=f"Оброблене зображення ({input_size})",
                    width=200
                )

                # Основний результат
                predicted_idx = st.session_state['predicted_class']
                st.markdown(f"""
                    <div style='background-color: #d4edda; padding: 20px; border-radius: 10px; border: 2px solid #28a745;'>
                        <h2 style='color: #155724; margin: 0;'>🎯 Передбачений клас</h2>
                        <h1 style='color: #155724; margin: 10px 0;'>{class_names[predicted_idx]}</h1>
                        <h3 style='color: #155724; margin: 0;'>{class_names_ukr[predicted_idx]}</h3>
                        <h2 style='color: #155724; margin-top: 10px;'>Впевненість: {st.session_state['confidence']:.2f}%</h2>
                    </div>
                """, unsafe_allow_html=True)

                st.markdown("---")

                # Графік ймовірностей
                st.subheader("📊 Розподіл ймовірностей")

                fig, ax = plt.subplots(figsize=(10, 8))

                # Кольори для графіка
                colors = ['#28a745' if i == predicted_idx else '#17a2b8' for i in range(10)]

                # Створення горизонтального графіка
                y_pos = np.arange(len(class_names))
                bars = ax.barh(y_pos, st.session_state['predictions'] * 100, color=colors, alpha=0.8, edgecolor='black')

                ax.set_yticks(y_pos)
                ax.set_yticklabels([f"{class_names[i]}\n({class_names_ukr[i]})" for i in range(10)], fontsize=10)
                ax.set_xlabel('Ймовірність (%)', fontsize=12, fontweight='bold')
                ax.set_title('Ймовірності класифікації для всіх класів', fontsize=14, fontweight='bold', pad=20)
                ax.grid(axis='x', alpha=0.3, linestyle='--')
                ax.set_xlim([0, 105])

                # Додавання значень на графік
                for i, (bar, prob) in enumerate(zip(bars, st.session_state['predictions'])):
                    width = bar.get_width()
                    label = f'{prob * 100:.2f}%'
                    ax.text(width + 1, bar.get_y() + bar.get_height() / 2,
                            label, ha='left', va='center', fontsize=10, fontweight='bold')

                plt.tight_layout()
                st.pyplot(fig)

                # Детальна таблиця
                st.subheader("📋 Детальна таблиця ймовірностей")

                import pandas as pd

                results_data = []
                for i in range(10):
                    results_data.append({
                        '№': i,
                        'Клас (English)': class_names[i],
                        'Клас (Українська)': class_names_ukr[i],
                        'Ймовірність (%)': f"{st.session_state['predictions'][i] * 100:.2f}",
                        'Передбачення': '✅ Так' if i == predicted_idx else '❌ Ні'
                    })

                df = pd.DataFrame(results_data)
                st.dataframe(
                    df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "№": st.column_config.NumberColumn("№", width="small"),
                        "Ймовірність (%)": st.column_config.TextColumn("Ймовірність (%)", width="medium"),
                    }
                )
            else:
                st.info("👆 Завантажте зображення та натисніть кнопку 'Класифікувати' для отримання результатів")

    # ============ ВКЛАДКА 3: АРХІТЕКТУРА ============
    with tab3:
        st.header(f"🔍 Архітектура моделі - {model_name}")

        # Виведення архітектури моделі
        st.subheader("📐 Структура нейронної мережі")

        string_buffer = io.StringIO()
        model.summary(print_fn=lambda x: string_buffer.write(x + '\n'))
        model_summary = string_buffer.getvalue()

        st.code(model_summary, language='text')

        st.markdown("---")

        # Опис класів
        st.subheader("📚 Класи Fashion MNIST")

        col1, col2 = st.columns(2)

        for i, (eng, ukr) in enumerate(zip(class_names, class_names_ukr)):
            if i < 5:
                col1.markdown(f"**{i}.** {eng} - *{ukr}*")
            else:
                col2.markdown(f"**{i}.** {eng} - *{ukr}*")

        st.markdown("---")

        # Порівняння моделей
        st.subheader("⚖️ Порівняння моделей")

        comparison_data = {
            'Характеристика': [
                'Архітектура',
                'Точність (Test)',
                'Кількість параметрів',
                'Час навчання',
                'Розмір входу',
                'Тип входу',
                'Переваги',
                'Недоліки'
            ],
            'Згорткова нейромережа (CNN)': [
                '3 Conv2D + 2 MaxPooling',
                '~91.83%',
                '93,322',
                '~3 хв 35 с',
                '28×28',
                'Grayscale',
                'Швидка, легка, ефективна',
                'Менша точність на складних даних'
            ],
            'VGG16 Transfer Learning': [
                'VGG16 + Custom layers',
                '~88.0%',
                '~15M',
                '~1 год 25 хв',
                '32×32',
                'RGB',
                'Потужні ознаки, transfer learning',
                'Повільніше навчання, більше ресурсів'
            ]
        }

        import pandas as pd

        df_comparison = pd.DataFrame(comparison_data)
        st.dataframe(df_comparison, use_container_width=True, hide_index=True)

else:
    st.error("""
    ❌ **Не вдалося завантажити модель!**

    Переконайтеся, що файли моделей знаходяться в тій самій директорії, що й app.py:
    - `fashion_mnist_cnn.keras`
    - `fashion_mnist_vgg16.keras`
    - `history_cnn.json`
    - `history_vgg16.json`
    """)

# Нижній колонтитул
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p><strong>Fashion MNIST Класифікатор</strong> | Створено для порівняння CNN та VGG16</p>
    <p>🛠️ Технології: TensorFlow 2.20 • Keras • Streamlit</p>
    <p>📊 Dataset: Fashion MNIST (60,000 train + 10,000 test images)</p>
</div>
""", unsafe_allow_html=True)