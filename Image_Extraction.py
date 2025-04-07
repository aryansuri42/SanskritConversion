import pytesseract
from PIL import Image, ImageOps

img = Image.open(r"A:\Projects\SanskritConversion\Sanskrit_Text_Conversion\images\Screenshot 2025-03-30 064158.jpg")
img = ImageOps.grayscale(img)
text = pytesseract.image_to_string(img, lang="extraction")
print(text)
"""९\ श्रदौी वाद ्रासीतस्वाद बरए सद्धंमासीत्‌स
इवादः सयमीश्वरखसप्रादाद ईश्रेए सहासीत्‌ तेन
स्नैवसुसरुजे सर्वषुखुषवसुपुकिमपिवसु तेनाखुषटं नकसि
#स जीवनदाकरः तत् जीवनं मनुबाणं चतिः; तज्ये
१तिरन्कारे मके किलनकारसन नग्राद।"""

"अथ्रनपणीतोतरम"

"नैकायाः तथैव नौकाकर्मचारिणं साहायं कर्तव्यम्‌ कदापि यदिनौका आपद्रस्ता"