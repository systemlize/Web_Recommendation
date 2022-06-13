import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("chonburi.csv")


df.columns = ['Timestamp', 'เพศ', 'อายุ', 'การศึกษา', 'อาชีพ', 'รายได้',
       'สถานภาพ',
       'เที่ยวบ่อย',
       'เที่ยวกับ',
       'ช่วงเวลา',
       'ประเภทสถานที่',
       'การใช้เงิน',
       'เกาะแสมสาร อ.สัตหีบ (Samaesan Island, Sattahip District)',
       'เเขาระเบิด ต.หนองใหญ่ อ.หนองใหญ่ (Khao Ra Bert, Nong Yai District)',
       'น้ำตกชันตาเถร ต.บางพระ อ.ศรีราชา (Chan Ta Than Waterfall, Si Racha District)',
       'แกรนด์แคนยอนคีรี อ.เมือง (Grand Canyon Khiri, Muang District)',
       'สวนนงนุช ต.นาจอมเทียน อ.สัตหีบ (Nong Nooch Tropical Garden, Sattahip District)',
       'เกาะล้าน ต.นาเกลือ อ.บางละมุง (Koh Larn, Bang Lamung District)',
       'ปราสาทสัจธรรม ต.นาเกลือ อ.บางละมุง (Sanctuary of Truth, Bang Lamung District)',
       'ชุมชนตะเคียนเตี้ย อ.บางละมุง (Takhian Tia Community, Bang Lamung District)',
       'วัดญาณสังวรารามวรมหาวิหาร ในพระบรมราชูปถัมภ์ ต.ห้วยใหญ่ อ.บางละมุง (Wat Yannasangwararam Woramahawihan, Bang Lamung District)',
       'เขาชีจรรย์ ต.นาจอมเทียน อ.สัตหีบ (Khao Chi Chan, Sattahip District)',
       'ถ้ำลอดมหาจักรพรรดิ์ ต.หนองปรือ อ.พนัสนิคม (Lod Maha Chakraphat Cave, Phanat Nikhom District)',
       'ตลาดน้ำสี่ภาคพัทยา ถ.สุขุมวิทพัทยา อ.บางละมุง (Pattaya Floating Market, Bang Lamung District)',
       'สวนน้ำ Cartoon Network Amazone ต.นาจอมเทียน อ.สัตหีบ (Cartoon Network Amazone Water Park, Sattahip District)',
       'Pattaya Shooting Park and Adventure สนามยิงปืนพัทยาชู้ทติ้งปาร์ค ต.ห้วยใหญ่ อ.บางละมุง (Pattaya Shooting Park and Adventure, Bang Lamung District)',
       'พิพิธภัณฑ์ริบลีส์ Ripley’s Believe It or Not อ.บางละมุง (Ripley’s Believe It or Not Museum, Bang Lamung District)',
       'Papa Beach Pattaya ต.นาจอมเทียน อ.สัตหีบ (Papa Beach Café Pattaya, Sattahip District)',
       'ทิฟฟานี่โชว์ อ.บางละมุง (Tiffany Show, Bang Lamung District)',
       'Austin café ต.อ่างศิลา อ.เมือง (Austin café, Ang Sila Subdistrict, Mueang District)',
       '331 bar & coffee war ต.พลูตาหลวง อ.สัตหีบ (331 bar & coffee war, Sattahip District)',
       'พิพิธภัณฑ์เกาะและทะเลไทย ต.แสมสาร อ.สัตหีบ (Thai Island and Sea Museum, Sattahip District)',
       'สถาบันวิทยาศาสตร์ทางทะเล มหาวิทยาลัยบูรพา ต.แสนสุข อ.เมือง (institute of marine science Burapha University, Mueang District)',
       'พิพิธภัณฑ์ภาพจิตกรรม 3 มิติ อ.บางละมุง (3D Art Museum, Bang Lamung District)']



df.drop('Timestamp', axis='columns', inplace=True)


df_person = df[['เพศ', 'อายุ', 'การศึกษา', 'อาชีพ', 'รายได้','สถานภาพ','เที่ยวบ่อย','เที่ยวกับ','ช่วงเวลา','ประเภทสถานที่','การใช้เงิน']]

df_place = df[['เกาะแสมสาร อ.สัตหีบ (Samaesan Island, Sattahip District)',
       'เเขาระเบิด ต.หนองใหญ่ อ.หนองใหญ่ (Khao Ra Bert, Nong Yai District)',
       'น้ำตกชันตาเถร ต.บางพระ อ.ศรีราชา (Chan Ta Than Waterfall, Si Racha District)',
       'แกรนด์แคนยอนคีรี อ.เมือง (Grand Canyon Khiri, Muang District)',
       'สวนนงนุช ต.นาจอมเทียน อ.สัตหีบ (Nong Nooch Tropical Garden, Sattahip District)',
       'เกาะล้าน ต.นาเกลือ อ.บางละมุง (Koh Larn, Bang Lamung District)',
       'ปราสาทสัจธรรม ต.นาเกลือ อ.บางละมุง (Sanctuary of Truth, Bang Lamung District)',
       'ชุมชนตะเคียนเตี้ย อ.บางละมุง (Takhian Tia Community, Bang Lamung District)',
       'วัดญาณสังวรารามวรมหาวิหาร ในพระบรมราชูปถัมภ์ ต.ห้วยใหญ่ อ.บางละมุง (Wat Yannasangwararam Woramahawihan, Bang Lamung District)',
       'เขาชีจรรย์ ต.นาจอมเทียน อ.สัตหีบ (Khao Chi Chan, Sattahip District)',
       'ถ้ำลอดมหาจักรพรรดิ์ ต.หนองปรือ อ.พนัสนิคม (Lod Maha Chakraphat Cave, Phanat Nikhom District)',
       'ตลาดน้ำสี่ภาคพัทยา ถ.สุขุมวิทพัทยา อ.บางละมุง (Pattaya Floating Market, Bang Lamung District)',
       'สวนน้ำ Cartoon Network Amazone ต.นาจอมเทียน อ.สัตหีบ (Cartoon Network Amazone Water Park, Sattahip District)',
       'Pattaya Shooting Park and Adventure สนามยิงปืนพัทยาชู้ทติ้งปาร์ค ต.ห้วยใหญ่ อ.บางละมุง (Pattaya Shooting Park and Adventure, Bang Lamung District)',
       'พิพิธภัณฑ์ริบลีส์ Ripley’s Believe It or Not อ.บางละมุง (Ripley’s Believe It or Not Museum, Bang Lamung District)',
       'Papa Beach Pattaya ต.นาจอมเทียน อ.สัตหีบ (Papa Beach Café Pattaya, Sattahip District)',
       'ทิฟฟานี่โชว์ อ.บางละมุง (Tiffany Show, Bang Lamung District)',
       'Austin café ต.อ่างศิลา อ.เมือง (Austin café, Ang Sila Subdistrict, Mueang District)',
       '331 bar & coffee war ต.พลูตาหลวง อ.สัตหีบ (331 bar & coffee war, Sattahip District)',
       'พิพิธภัณฑ์เกาะและทะเลไทย ต.แสมสาร อ.สัตหีบ (Thai Island and Sea Museum, Sattahip District)',
       'สถาบันวิทยาศาสตร์ทางทะเล มหาวิทยาลัยบูรพา ต.แสนสุข อ.เมือง (institute of marine science Burapha University, Mueang District)',
       'พิพิธภัณฑ์ภาพจิตกรรม 3 มิติ อ.บางละมุง (3D Art Museum, Bang Lamung District)']]


df_new_user = pd.read_csv('chonburi_new_user.csv')

df_new_user.columns = ['เพศ', 'อายุ', 'การศึกษา', 'อาชีพ', 'รายได้',
       'สถานภาพ',
       'เที่ยวบ่อย',
       'เที่ยวกับ',
       'ช่วงเวลา',
       'ประเภทสถานที่',
       'การใช้เงิน',]

col_names = ['เพศ', 'อายุ', 'การศึกษา', 'อาชีพ', 'รายได้','สถานภาพ','เที่ยวบ่อย','เที่ยวกับ','ช่วงเวลา','ประเภทสถานที่','การใช้เงิน']

dummies_df_person = pd.get_dummies(df_person[col_names])


df_person = pd.concat([df_person, dummies_df_person], axis=1)

df_person = df_person.drop(col_names, axis=1)


dummies_df_new_user = pd.get_dummies(df_new_user[col_names])

df_new_user = pd.concat([df_new_user, dummies_df_new_user], axis=1)

df_new_user = df_new_user.drop(col_names, axis=1)


df_all = df_person.append(df_new_user, ignore_index=True, sort=False)


df_all = df_all.fillna(0)


x_user = cosine_similarity(df_all)


def travel_reccomender(df_all, df_place, x_user, user_ix=-1, k=403, top_n=10):
    user_similarities = x_user[user_ix]

    most_similar_users = df_all.index[user_similarities.argpartition(-k)[-k:]]
    most_similar_users = most_similar_users[:-1]


    rec_place = df_place.iloc[most_similar_users].mean(0).sort_values(ascending=False)
    rec_place_top = rec_place.head(top_n)
    print(rec_place_top)

    return rec_place_top


travel_reccomender(df_all, df_place, x_user, user_ix=-1, k=5, top_n=5)