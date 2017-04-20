import logging
import os
import cloudstorage as gcs
import webapp2

from google.appengine.api import app_identity


#Specifying the Cloud Storage bucket
def get(self):
	bucket_name = os.environ.get('BUCKET_NAME',app_identity.get_default_gcs_bucket_name())

	self.response.headers['Content-Type'] = ('text/plain')
	self.response.write('Demo GCS Application running from Version: '
						+ os.environ['CURRENT_VERSION_ID'] + '\n')
	self.response.write('Using bucket name: ' + bucket_name + '\n\n')

#Reading from Cloud Storage
def read_file(self, filename):
  
  self.response.write('Abbreviated file content (first line and last 1K):\n')

  gcs_file = gcs.open(filename)
  self.response.write(gcs_file.readline())
  gcs_file.seek(-1024, os.SEEK_END)
  self.response.write(gcs_file.read())
  gcs_file.close()
