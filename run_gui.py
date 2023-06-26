import matplotlib.pyplot as plt
import torch
from PIL import Image
import requests
from io import BytesIO
from rotate_captcha_crack.common import device
from rotate_captcha_crack.model import RotNetR, WhereIsMyModel
from rotate_captcha_crack.utils import process_captcha
import pyperclip
import tkinter as tk


def getDistance(imageUrl):
    # Your Python function code here
    with torch.no_grad():
        model = RotNetR(train=False, cls_num=180)
        model_path = WhereIsMyModel(model).with_index(-1).model_dir / "best.pth"
        print(f"Use model: {model_path}")
        model.load_state_dict(torch.load(str(model_path), map_location=torch.device('cpu')))  # EDITED
        model = model.to(device=device)
        model.eval()

        # Get the image data from the URL
        img_data = requests.get(imageUrl).content

        # Load the image from the image data
        img = Image.open(BytesIO(img_data))

        # Display the image
        #img.show()  # FOR DEBUGGING ONLY

        img_ts = process_captcha(img)
        img_ts = img_ts.to(device=device)

        predict = model.predict(img_ts)
        degree = predict * 360

    return round(827 + 205 * predict)


def submit():
    # Get the value from the input field
    input_value = pyperclip.paste()

    # Call the function with the input value
    output_value = getDistance(input_value)

    # Update the output text field with the result
    output_field.config(state=tk.NORMAL)
    output_field.delete(1.0, tk.END)
    output_field.insert(tk.END, output_value)
    output_field.config(state=tk.DISABLED)


def copy_to_clipboard():
    # Get the value from the output field
    output_value = output_field.get(1.0, tk.END)

    # Copy the value to the clipboard
    pyperclip.copy(output_value)


# Create the main window
root = tk.Tk()


# Create the submit button
submit_button = tk.Button(root, text="Get URL From Clipboard", command=submit)
submit_button.pack()

# Create the output text field
output_field = tk.Text(root, height=5, state=tk.DISABLED)
output_field.pack()

# Create the copy button
copy_button = tk.Button(root, text="Copy to Clipboard", command=copy_to_clipboard)
copy_button.pack()

# Start the main event loop
root.mainloop()
