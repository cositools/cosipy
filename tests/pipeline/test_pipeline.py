import os
from cosipy import test_data
#
def replace_text_in_file(file_path, old_text, new_text, file_newpath):
    try:
        with open(file_path, 'r') as file:
            file_data = file.read()

        # Replace the old text with the new text
        updated_data = file_data.replace(old_text, new_text)

        with open(file_newpath, 'w+') as file:
            file.write(updated_data)

        print(f"Text replaced successfully in {file_path}")
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
#
data_path = str(test_data.path)
config_tmp_path=str(data_path+"/test_spec_grb_tmp.yaml")
config_path=str(data_path+"/test_spec_grb.yaml")
replace_text_in_file(config_tmp_path,"TDATAPATH",data_path,config_path)
#
def test_output_ex():
    os.system(str("cosi-threemlfit --config "+ config_path))
    assert os.path.exists(str(data_path+"/grb_test.fits"))
    assert os.path.exists(str(data_path+"/fit_grb_test.pdf"))