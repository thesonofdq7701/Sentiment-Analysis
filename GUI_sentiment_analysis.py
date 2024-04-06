import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import load
from function_preprocessing import optimized_process_text, sentiment_score

# Using menu
st.title("DATA SCIENCE PROJECT")
menu = ["Introduction", "My Project", "Predict New Data"]
choice = st.sidebar.selectbox('Danh mục', menu)

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
reviews_df_raw = pd.read_csv("2_Reviews.csv")
restaurants_df_raw = pd.read_csv("1_Restaurants.csv")
reviews_df = pd.read_csv("processed_reviews.csv")
model_SVM = load('svm_project1.joblib')
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

if choice == 'Introduction':    
    st.subheader("GIỚI THIỆU")  
    st.subheader("Sentiment Analysis") 
    # Hiển thị hình ảnh từ file
    image = open("images/sentiment_analysis.png", "rb").read()
    st.image(image, use_column_width=True)
    
    st.write("## I. Mục tiêu")  
    st.write("""### **Mục tiêu của Sentiment Analysis là phân tích và hiểu cảm xúc hoặc ý kiến được thể hiện trong review nhà hàng của các khách hàng trước từ đó xác định xem khách hàng thể hiện ý kiến tích cực, tiêu cực hoặc trung lập về nhà hàng đó.**""")
    st.write("## II. Mô hình SVM trong Sentiment Analysis")
    st.write("""### **Trong bài toán Sentiment Analysis, mô hình SVM (Support Vector Machine) được sử dụng để phân loại các đoạn văn bản thành các loại cảm xúc khác nhau như tích cực, tiêu cực hoặc trung tính.**
### **Nói một cách đơn giản, thuật toán SVM giúp chúng ta tổ chức dữ liệu văn bản thành các loại cảm xúc khác nhau dựa trên các đặc trưng của chúng, như từ ngữ hoặc cấu trúc câu. Điều này giúp chúng ta hiểu được cảm xúc chung của các đoạn văn bản của khách hàng một cách tự động và hiệu quả.**
""")
    st.write("## III. Dữ liệu")
    # Hiển thị 10 dòng đầu của dữ liệu
    st.write("### **Đây là dữ liệu đánh giá của một vài khách hàng:**")
    st.dataframe(reviews_df_raw.head(5))
    st.write("### **Đây là dữ liệu của một vài nhà hàng:**")
    st.dataframe(restaurants_df_raw.head(5))

    st.write("## IV. Nhóm")
    st.write("### - Nguyễn Thế Sơn")
    st.write("### - Nguyễn Trung Sơn")

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
elif choice == 'My Project':    

    st.subheader("I.Giới thiệu về dữ liệu")
    st.write("### **Dữ liệu gốc:**")
    st.dataframe(reviews_df_raw.sample(3))
    st.dataframe(restaurants_df_raw.sample(3))
    st.write("### **Dữ liệu tổng hợp:**")
    st.dataframe(reviews_df.sample(3))

    st.subheader("II.Trực quan hóa dữ liệu")
    st.write("### **1.Rating của khách hàng**")
    image = open("images/rating_distribution.png", "rb").read()
    st.image(image, use_column_width=True)
    st.write("""### **Nhận xét:**
#### - **Thang điểm về dịch vụ của các khách hàng từ 0 đến 10 điểm và chủ yếu các khách hàng đánh giá khoảng từ 6 đến 10 điểm.**""")
    st.write("### **2.Các nhà hàng theo quận**")
    image = open("images/Count_restaurants_in_each_disstrict.png", "rb").read()
    st.image(image, use_column_width=True)
    image = open("images/mean_rating_by_district.png", "rb").read()
    st.image(image, use_column_width=True)
    st.write("""### **Nhận xét:**
#### - **Các nhà hàng chủ yếu tập trung ở Quận 1 đến Quận 5. Đây là nơi tập trung đông dân cư, nhà hàng, khách sạn. Trung tâm vui chơi giải trí cũng chủ yếu tập trung ở các quận này.**
#### - **Thang điểm trung bình của các nhà hàng tại mỗi quận khá đều.**""")
    
    st.subheader("III.Customer Segmentation")
    st.write("### **1.Kết quả**")
    st.dataframe(reviews_df.head(10))
    st.write("### **2.Trực quan hóa**")
    image = open("images/Count_sentment.png", "rb").read()
    st.image(image, use_column_width=True)
    st.write("""### **Nhận xét:**
#### - **Chủ yếu các nhận xét là tích cực, các nhận xét tiêu cực và trung tính ở mức trung bình. Tuy nhiên đánh giá của nhiều khách hàng không khớp với lời nhận xét của họ có lẽ vì họ đang nhầm thang điểm.**
""")
    st.write("""### **Kết luận:**
#### - **Ta nên sử dụng comment của khách hàng để đánh giá thay vì dùng rating.**           
""")

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


elif choice=='Predict New Data':
    st.subheader("Predict Comment")
    # Cho người dùng chọn nhập dữ liệu hoặc upload file
    type = st.radio("Chọn cách nhập dữ liệu", options=["Nhập dữ liệu vào text area", "Nhập nhiều dòng dữ liệu trực tiếp", "Upload file"])
    # Nếu người dùng chọn nhập dữ liệu vào text area
    if type == "Nhập dữ liệu vào text area":
        st.subheader("Nhập dữ liệu vào text area")
        content = st.text_area("#### **Nhập ý kiến:**")
        # Nếu người dùng nhập dữ liệu đưa content này vào thành 1 dòng trong DataFrame
        if content:
            preprocessing_content = optimized_process_text(content)
            preprocessing_content = pd.Series(preprocessing_content)
            content = pd.Series([content])
            predict_data = pd.DataFrame({'Sentiment':model_SVM.predict(preprocessing_content)})
            df = pd.concat([content, predict_data], axis=1)
            first_column_name = df.columns[0]
            df.rename(columns={first_column_name:'Comment'}, inplace=True)
        submitted_project1 = st.button("#### **Dự đoán ý kiến:**")
        if submitted_project1:
            st.dataframe(df)
    # Nếu người dùng chọn nhập nhiều dòng dữ liệu trực tiếp vào một table
    elif type == "Nhập nhiều dòng dữ liệu trực tiếp":
        st.subheader("Nhập nhiều dòng dữ liệu trực tiếp")        
        preprocessing_comments = []
        comments = []
        sentiment = []
        for i in range(5):
            comment = st.text_area(f"Nhập ý kiến {i+1}:")
            comments.append({"Comment": comment})
            preprocessing_comment = optimized_process_text(comment)
            preprocessing_comments.append({"Comment": preprocessing_comment})

        # Chuyển đổi danh sách thành DataFrame
        comments_df = pd.DataFrame(comments)
        preprocessing_comments_df = pd.DataFrame(preprocessing_comments)

        # Sử dụng model_SVM để dự đoán cảm xúc và tạo DataFrame mới cho dự đoán
        predictions = model_SVM.predict(preprocessing_comments_df['Comment'])
        predicted_sentiments = pd.DataFrame({"Sentiment": predictions})
        results = pd.concat([comments_df, predicted_sentiments], axis=1)
        submitted_project1 = st.button("#### **Hiển thị kết quả dự đoán cảm xúc...**")
        if submitted_project1:
            st.dataframe(results)

             
    # Nếu người dùng chọn upload file
    elif type == "Upload file":
        st.subheader("Upload file")
        # Upload file
        uploaded_file = st.file_uploader("Chọn file dữ liệu", type=["csv", "txt"])
        if uploaded_file is not None:
            # Đọc file dữ liệu
            df = pd.read_csv(uploaded_file)
            df['Processed_Comment'] = df['Comment'].apply(lambda x: optimized_process_text(x))
            predictions = model_SVM.predict(df['Processed_Comment'])
            st.write(df)
            predictions_df = pd.DataFrame(predictions)
            
    # Từ df này, người dùng có thể thực hiện các xử lý dữ liệu khác nhau
    submitted_project1 = st.button("Dự đoán cảm xúc")
    if submitted_project1:
        st.write("Hiển thị kết quả dự đoán cảm xúc...")
        # result = pd.concat([df['Comment'], predictions_df], axis = 1)
        st.write(df)
