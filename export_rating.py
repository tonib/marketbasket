from export_candidates import export_model
from datetime import datetime

print(datetime.now(), "Process start: Export rating model")
export_model(True)
print(datetime.now(), "Process end: Export rating model")
