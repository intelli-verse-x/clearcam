"""Notification protocol helpers; importing this module sends nothing."""
import http.client
import os
import uuid


def should_send_notification(settings, vod, is_notification):
    return bool(settings.use_notifs() and not vod and is_notification)


def send_notif(session_token: str, text=None, body_text=None, host="https://clearcam.org", img=None):
    https = host.startswith("https://")
    if host.startswith("http"): host = host[host.index("//")+2:] # remove https:// / http:
    endpoint = "/send" #/test
    boundary = f"Boundary-{uuid.uuid4()}"
    content_type = f"multipart/form-data; boundary={boundary}"
    lines = []
    if session_token is not None:
      lines.extend([
        f"--{boundary}",
        'Content-Disposition: form-data; name="session_token"',
        "",
        session_token,
      ])
    if text is not None:
      lines.extend([
        f"--{boundary}",
        'Content-Disposition: form-data; name="text"',
        "",
        text,
      ])
    if body_text is not None:
      lines.extend([
        f"--{boundary}",
        'Content-Disposition: form-data; name="body_text"',
        "",
        body_text,
      ])

    body = ("\r\n".join(lines) + ("\r\n" if lines else "")).encode("utf-8")
    if img is not None and host != "clearcam.org":
      with open(img, "rb") as f: img_data = f.read()
      body += (f"--{boundary}\r\n"f'Content-Disposition: form-data; name="img"; filename="{os.path.basename(img)}"\r\n'f"Content-Type: image/jpeg\r\n\r\n").encode() + img_data + b"\r\n"

    body += f"--{boundary}--\r\n".encode()
    conn = http.client.HTTPSConnection(host) if https else  http.client.HTTPConnection(host)
    headers = {"Content-Type": content_type, "Content-Length": str(len(body))}
    try:
        conn.request("POST", endpoint, body, headers)
        response = conn.getresponse()
        print(f"Status: {response.status} {response.reason}")
        print(response.read().decode())
    except Exception as e:
        print(f"Error sending session token: {e}")
    finally:
        conn.close()

