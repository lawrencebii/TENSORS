import urllib.request
import zipfile

validation_file = "validation-horse-or-human.zip"
validation_url = "https://storage.googleapis.com/laurencemoroney-blog.appspot.com/validation-horse-or-human.zip"
validation_dir = 'horse-or-human/validation/'
urllib.request.urlretrieve(validation_url,validation_file)
zip_ref = zipfile.ZipFile(validation_file,'r')
zip_ref.extractall(validation_dir)
zip_ref.close()