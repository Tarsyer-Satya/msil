import cv2
import numpy as np
def putText(text_arr,img):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.4
    font_color = (255, 255, 255)  # White color
    font_thickness = 5
    height, width = img.shape[:2]
    w_off = 10
    h_off = 10
    next_text_off = 0.05
    for ind, text in enumerate(text_arr):
        print(text)
        cv2.rectangle(img, (int(width*0.83) - w_off , int(height*(1-0.95)) + h_off + int(ind*height*next_text_off)), (int(width*1) + w_off, int(height*(1-0.98)) - h_off + int(ind*height*next_text_off)), (0,255,0), -1)
        cv2.putText(img, str(text), (int(width*0.83), int(height*(1-0.95)) + int(ind*height*next_text_off)), font, font_scale, font_color, font_thickness)
    return img


class Find_Bolts:
    def __init__(self,image_path = '',image = [], roi=[], inv_mask = False):
        self.image_path = image_path
        if image_path == "":
            self.image = image
        else:
            self.image = cv2.imread(image_path)
        
        self.inv_mask = inv_mask
        self.roi = roi
        # self.roi_image = roi_2

    def show_image(self):
        cv2.imshow('roi_image', self.roi_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    def show_image_with_bb(self,bb,image = []):
        if(len(image) == 0):
            image = self.image
        cv2.polylines(image,[bb], isClosed = True, color=(255,255,0), thickness= 2)
        cv2.imshow('Image with Polylines', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return image
    def apply_effects(self, blur = False, smooth = False, threshold = False, erosion = False, dilation = False,mask_inv = False):
        effect_image = self.image
        mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
        # if mask_inv : mask = cv2.bitwise_not(mask)
        cv2.drawContours(mask, [self.roi], -1, (255), -1)
        
        effect_image = cv2.bitwise_and(self.image, self.image, mask=mask)
        


        if blur:
            effect_image = cv2.GaussianBlur(effect_image, (5, 5), 0)
        if smooth:
            effect_image = cv2.medianBlur(effect_image, 5)
        if threshold:
            gray_image = cv2.cvtColor(effect_image, cv2.COLOR_BGR2GRAY)
            _, effect_image = cv2.threshold(gray_image, 130, 255, cv2.THRESH_BINARY)
        if erosion:
            kernel = np.ones((5,5), np.uint8) 
            effect_image = cv2.erode(effect_image, kernel, iterations=1)
        if dilation:    
            kernel = np.ones((5,5), np.uint8)  # Define a 5x5 rectangular kernel
            effect_image = cv2.dilate(effect_image, kernel, iterations=5)    
        
        self.contour_image = effect_image
        # cv2.imshow("final Image",self.contour_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    
    def find_contours(self):
        contours, _ = cv2.findContours(self.contour_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return len(contours)



class Find_Boxes(Find_Bolts):
    def __init__(self,image_path = '', image = [],roi = [], inv_mask = False):
        super().__init__(image_path = image_path,image = image,roi = roi, inv_mask = inv_mask)

    def find_contours(self):
        binary = cv2.bitwise_not(self.contour_image)
        

        boxes_count = 0
        card_boxes_count = 0
        card_values = []
        empty_values = []
        

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Define threshold for object area
        threshold_object_area = 30000  # Adjust as needed
        # print(len([True for contour in contours if cv2.contourArea(contour) > 30000]))

        # # Iterate through each contour
        for contour in contours:
            # Calculate area of the contour (object)
            area = cv2.contourArea(contour)

            # print("box area:", area)
            
            # If the area exceeds the threshold, process the contour
            if area > threshold_object_area:
                # Draw the contour (object) on a black canvas
                boxes_count += 1
                object_mask = np.zeros_like(binary)
                cv2.drawContours(object_mask, [contour], -1, 255, thickness=cv2.FILLED)
                
                # Invert the object mask to isolate the white areas (cards) inside the object
                inverted_object_mask = cv2.bitwise_not(object_mask)
                
                # Bitwise AND the inverted object mask with the binary image to extract the cards
                cards_inside_object = cv2.bitwise_and(binary, binary, mask=inverted_object_mask)
                
                # Find contours of the white areas (cards) inside the object
                cards_contours, _ = cv2.findContours(cards_inside_object, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Iterate through each card contour and calculate its area
                for card_contour in cards_contours:
                    card_area = cv2.contourArea(card_contour)
                    if(card_area >= 12000 and card_area <= 20000):
                        card_boxes_count += 1
                        # print("Area of card inside object:", card_area)
                        

                    
                

        return {
            "card_boxes_count":card_boxes_count,
            "boxes_count":boxes_count
        }


        return [internal_contours_with_cards]
    # Function to check if contour contains white part
    def contour_black_pixel_count(self, contour, binary_image):
        # Create a mask for the contour
        mask = np.zeros_like(binary_image)
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
        
        # Bitwise AND operation to extract contour region from binary image
        contour_region = cv2.bitwise_and(binary_image, mask)
        
        # Count the number of white pixels (value 255) within the contour region
        white_pixel_count = np.count_nonzero(contour_region == 255)
        
        return white_pixel_count

    def apply_effects(self, blur = False, smooth = False, threshold = False, erosion = False, dilation = False,mask_inv = False):
        effect_image = self.image
        mask = np.ones(self.image.shape[:2], dtype=np.uint8) * 255  # Initialize mask with all white pixels
        cv2.drawContours(mask, [self.roi], -1, (0), -1)  # Draw contour filled with black on the white mask

        

        effect_image = cv2.bitwise_and(self.image, self.image, mask=cv2.bitwise_not(mask)) 
        effect_image[mask == 255] = (255, 255, 255) 
        
        


        if blur:
            effect_image = cv2.GaussianBlur(effect_image, (5, 5), 0)
        if smooth:
            effect_image = cv2.medianBlur(effect_image, 5)
        if threshold:
            gray_image = cv2.cvtColor(effect_image, cv2.COLOR_BGR2GRAY)
            _, effect_image = cv2.threshold(gray_image, 130, 255, cv2.THRESH_BINARY)
        if erosion:
            kernel = np.ones((5,5), np.uint8) 
            effect_image = cv2.erode(effect_image, kernel, iterations=1)
        if dilation:    
            kernel = np.ones((10,10), np.uint8)  # Define a 5x5 rectangular kernel
            effect_image = cv2.dilate(effect_image, kernel, iterations=5)    
        
        self.contour_image = effect_image
        # cv2.imshow("final Image",self.contour_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


class Gourmet_Fitment(Find_Boxes):
    def __init__(self,image_path = '', image = [],roi = [], inv_mask = False):
        super().__init__(image_path = image_path,image = image,roi = roi, inv_mask = inv_mask)

    
    def find_contours(self):
        binary = cv2.bitwise_not(self.contour_image)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)       

        return {
            "Number_of_Fitments":len(contours),   
        }
    
    
    
    
    def apply_effects(self, blur = False, smooth = False, threshold = False, erosion = False, dilation = False,mask_inv = False):
        effect_image = self.image
        mask = np.ones(self.image.shape[:2], dtype=np.uint8) * 255  # Initialize mask with all white pixels
        cv2.drawContours(mask, [self.roi], -1, (0), -1)  # Draw contour filled with black on the white mask

        

        effect_image = cv2.bitwise_and(self.image, self.image, mask=cv2.bitwise_not(mask)) 
        effect_image[mask == 255] = (255, 255, 255) 
        
        


        if blur:
            effect_image = cv2.GaussianBlur(effect_image, (5, 5), 0)
        if smooth:
            effect_image = cv2.medianBlur(effect_image, 5)
        if threshold:
            gray_image = cv2.cvtColor(effect_image, cv2.COLOR_BGR2GRAY)
            _, effect_image = cv2.threshold(gray_image, 40, 255, cv2.THRESH_BINARY)
        if erosion:
            kernel = np.ones((5,5), np.uint8) 
            effect_image = cv2.erode(effect_image, kernel, iterations=1)
        if dilation:    
            kernel = np.ones((5,5), np.uint8)  # Define a 5x5 rectangular kernel
            effect_image = cv2.dilate(effect_image, kernel, iterations=4)    
        
        self.contour_image = effect_image
        # cv2.imshow("final Image",self.contour_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()