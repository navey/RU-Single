from selenium import webdriver
from time import sleep
from PIL import Image
from io import BytesIO
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import *
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys

from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier

import cv2 as ocv
import dlib
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import face_recognition
import imutils
import sklearn

from credentials import username, password

class Python_Bot():
    def __init__(self):
        self.driver = webdriver.Chrome()
    
    def login(self):
        # Go to Tinder
        self.driver.get('https://tinder.com')

        sleep(2)

        # Click "Login with Facebook" button
        self.driver.implicitly_wait(20)
        fb_btn = self.driver.find_element_by_xpath('/html/body/div[2]/div/div/div/div/div[3]/span/div[2]/button')
        fb_btn.click()

        # Switch handles to popup login window
        base_window = self.driver.window_handles[0] #holds the handle for the Tinder handle
        self.driver.switch_to.window(self.driver.window_handles[1])

        # Fill in Facebook login information
        email_text = self.driver.find_element_by_xpath('//*[@id="email"]')
        email_text.send_keys('naveenan.r.y@gmail.com')        
        password_text = self.driver.find_element_by_xpath('//*[@id="pass"]')
        password_text.send_keys('ramanan36')

        # Login to Facebook
        login_btn = self.driver.find_element_by_xpath('//*[@id="u_0_0"]')
        login_btn.click()
        
        # Allow location and don't allow notifications
        self.driver.switch_to.window(base_window)
        location_btn = self.driver.find_element_by_xpath('//*[@id="modal-manager"]/div/div/div/div/div[3]/button[1]')
        location_btn.click()
        notification_btn = self.driver.find_element_by_xpath('/html/body/div[2]/div/div/div/div/div[3]/button[2]')
        notification_btn.click()
        """

        names = ['face-width', 'left-eyebrow', 'right-eyebrow', 'nose-length', 'nose-tip', 'lefteye-width', 'lefteye-length', 'righteye-width', 'righteye-length', 'toplip-length', 'mouth', 'right-swipe']
        df = pd.read_csv("outFile.csv", names=names)

        X = df.drop('right-swipe', axis=1)
        y = df['right-swipe']

        # implementing train-test-split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=66)

        # random forest model creation
        rfc = RandomForestClassifier()
        rfc.fit(X_train,y_train)
        # predictions
        rfc_predict = rfc.predict(X_test)

        count = 0
        sleep(10)

        while(count < 50):
            sleep(2)

            # Get the location and size of the image 
            image_path = self.driver.find_element_by_xpath('//*[@id="content"]/div/div[1]/div/main/div[1]/div/div/div[1]')
            image_location = image_path.location
            image_size = image_path.size
            png = self.driver.get_screenshot_as_png()

            # Get image and resize for cropping
            im = Image.open(BytesIO(png))
            screensize = (self.driver.execute_script("return document.body.clientWidth"), self.driver.execute_script("return window.innerHeight"))
            im = im.resize(screensize)

            # Set up cropping sizes
            left = image_location['x']
            top = image_location['y']
            right = image_location['x'] + image_size['width']
            bottom = image_location['y'] + image_size['height']

            # Crop and save the image
            im = im.crop((left, top, right, bottom))
            im.save('picture.png')

            # Analyze picture
            image = face_recognition.load_image_file('picture.png')
            lands = face_recognition.face_landmarks(image)
            try:
                total_width = (lands[0]['chin'][-1][0]-lands[0]['chin'][0][0])+15
                total_length = (lands[0]['chin'][8][0]-lands[0]['left_eyebrow'][2][0])+15
                attributes = [(lands[0]['chin'][-1][0]-lands[0]['chin'][0][0])/total_width, (lands[0]['left_eyebrow'][-1][0]-lands[0]['left_eyebrow'][0][0])/total_width, (lands[0]['right_eyebrow'][-1][0]-lands[0]['right_eyebrow'][0][0])/total_width, (lands[0]['nose_bridge'][-1][-1]-lands[0]['nose_bridge'][0][-1])/total_length, (lands[0]['nose_tip'][-1][0]-lands[0]['nose_tip'][0][0])/total_width, (lands[0]['left_eye'][-1][0]-lands[0]['left_eye'][0][0])/total_width, (lands[0]['left_eye'][-2][-1]-lands[0]['left_eye'][1][-1])/total_length, (lands[0]['right_eye'][-1][0]-lands[0]['right_eye'][0][0])/total_width, (lands[0]['right_eye'][-2][-1]-lands[0]['right_eye'][1][-1])/total_length, (lands[0]['top_lip'][6][0]-lands[0]['top_lip'][0][0])/total_width, (lands[0]['bottom_lip'][3][-1]-lands[0]['top_lip'][4][-1])/total_length]
                attribute_array = np.asarray(attributes)
            except:
                self.swipe_left(count, im)
                continue

            attribute_array = attribute_array.reshape(1,-1)
            # Go to next photo after comparing
            if rfc.predict(attribute_array) == 1:
                self.swipe_right(count, im)
            else:
                self.swipe_left(count, im)            

            count += 1

        self.driver.quit()
        """

    def swipe_left(self,count,im):
        ActionChains(self.driver).send_keys(Keys.LEFT).perform()
        im.save('left/picture' + str(count) + '.png')

    def swipe_right(self,count,im):
        ActionChains(self.driver).send_keys(Keys.RIGHT).perform()
        im.save('right/picture' + str(count) + '.png')    
        

if __name__=="__main__":
    bot = Python_Bot()
    bot.login()