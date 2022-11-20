import gspread
gc = gspread.service_account(filename = "creds.json")
spreadsheet_id = "13ZB78uxyCPzT4s8dpPwHrHYQa9_K8SweOeOCPOaWOr4"
sh = gc.open_by_key(spreadsheet_id).sheet1