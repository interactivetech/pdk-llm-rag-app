# Bring your own data

1. Assume you have a csv file that contains data:

```csv
Title,Date,Summary,Content,URL
The Impact of Superfoods,"February 15, 2024",Exploring the nutritional benefits and myths surrounding superfoods.,"Superfoods have gained popularity for their supposed health benefits, including improved energy levels and immune system support. However, it's important to approach the superfood trend with a critical eye, as not all claims are supported by scientific evidence. A balanced diet incorporating a variety of nutrients is essential for good health.",https://www.nutrition.org/superfoods-impact
Balancing Macros for Optimal Health,"February 20, 2024","How to balance carbohydrates, proteins, and fats for a healthier lifestyle.",Understanding the role of macronutrients in your diet is crucial for managing weight and maintaining muscle mass. This article breaks down the science behind macronutrients and provides tips for achieving a balanced diet that supports your health and fitness goals.,https://www.healthyliving.org/balancing-macros
Hydration: More Than Just Water,"March 1, 2024",Unveiling the importance of staying hydrated with more than just water.,"While water is essential for hydration, electrolytes like sodium, potassium, and magnesium play a critical role in maintaining fluid balance in the body. Learn about the sources of these important nutrients and how to stay properly hydrated.",https://www.wellnesswater.org/hydration-basics
Plant-based Diets: A Comprehensive Guide,"March 10, 2024",Exploring the health benefits and challenges of adopting a plant-based diet.,"Plant-based diets are associated with lower risks of heart disease, hypertension, diabetes, and certain types of cancer. This guide covers everything from nutritional considerations to delicious plant-based recipes to help you transition to a more sustainable diet.",https://www.veganhealth.org/plant-based-guide
Understanding Food Labels,"March 20, 2024",Decoding the information on food packaging to make healthier choices.,"Food labels can be confusing, with terms like ""natural,"" ""organic,"" and ""non-GMO"" often used ambiguously. This article explains what these labels mean and how to use the nutritional information to make informed decisions about the food you eat.",https://www.foodsafety.gov/label-reading

```
copy of file in `nutrition_content.csv` is in repo.

2. Run this code to convert to PDF named `custom_pdf.pdf`
```python
from fpdf import FPDF
import csv

# Define the CSV file name
csv_file = 'nutrition_content.csv'

# Create a PDF class instance
class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        # self.cell(0, 10, 'Document Generated from CSV', 0, 1, 'C')

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        # self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

# Create a PDF object
pdf = PDF()
pdf.add_page()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.set_font('Arial', '', 12)
# Specify the rows you want to include in the PDF
# For example, to include rows 1-10 and 20, use: rows_to_include = list(range(1, 11)) + [20]

# Open the CSV file and read contents
with open(csv_file, newline='', encoding='utf-8-sig') as csvfile:
    reader = csv.DictReader(csvfile)
    for i, row in enumerate(reader, start=1):
        title = row['Title'].encode('latin-1', 'replace').decode('latin-1')
        date = row['Date'].encode('latin-1', 'replace').decode('latin-1')
        content = row['Content'].encode('latin-1', 'replace').decode('latin-1')

        # Add content to PDF
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, title, 0, 1)
        pdf.set_font('Arial', 'I', 12)
        pdf.cell(0, 10, date, 0, 1)
        pdf.set_font('Arial', '', 12)
        pdf.multi_cell(0, 10, content)
        pdf.add_page()

# Save the PDF to a file
pdf.output('custom_pdf.pdf'.format(rows_to_include[0]))
```

3. Move the pdf to the `pdf_data` folder

4. Follow the same instructions in the Deploy RAG with PDK to upload and have LLM RAG process, index, and deploy data:
```bash
!pachctl create repo data

```bash
%%capture
# note xml contains folder of older press releases (2021-2022)
!pachctl put file data@master: -r -f data/HPE_press_releases/
```
```bash
# current press releases in csv format
!pachctl put file data@master: -r -f data/HPE_2023_Press_Releases.csv

```bash
!pachctl put file data@master: -f pdf_data/output.pdf
```