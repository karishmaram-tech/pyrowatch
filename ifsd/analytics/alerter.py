"""
================================================================================
  PyroWatch/analytics/alerter.py
  AlertMailer -- Gmail Email Notification System
================================================================================
  Sends an email alert with screenshot attachment when CRITICAL tier
  is triggered. Built-in cooldown prevents inbox flooding.

  REQUIRES:
    - A Gmail account with 2FA enabled
    - A Gmail App Password (16 characters)
    - Both stored in a .env file in the project root (never hardcoded)

  .env FILE FORMAT (create this file manually):
    PyroWatch_EMAIL_SENDER=yourname@gmail.com
    PyroWatch_EMAIL_PASSWORD=abcdefghijklmnop
    PyroWatch_EMAIL_RECEIVER=yourname@gmail.com

  SECURITY NOTES:
    - NEVER commit the .env file to git
    - NEVER hardcode credentials in Python files
    - The .env file is read once at startup and stored in memory only
================================================================================
"""

import os
import time
import datetime
import smtplib
import tempfile
from email.mime.multipart import MIMEMultipart
from email.mime.text      import MIMEText
from email.mime.image     import MIMEImage

import cv2
import numpy as np


class AlertMailer:
    """
    Sends Gmail email alerts when CRITICAL risk tier is detected.

    HOW TO USE:
        mailer = AlertMailer()

        # inside your frame loop after classify_risk():
        if tier == "CRITICAL":
            mailer.send_alert(
                tier       = tier,
                fire_conf  = fr["confidence"],
                smoke_conf = sr["confidence"],
                risk_score = score,
                frame_num  = frame_num,
                canvas     = canvas,   # the HUD-rendered frame
            )
        # send_alert() handles cooldown internally --
        # safe to call every frame, won't spam your inbox.
    """

    SMTP_HOST    = "smtp.gmail.com"
    SMTP_PORT    = 587
    ENV_FILE     = ".env"

    def __init__(
        self,
        cooldown_seconds: int = 60,
        min_tier_to_alert: str = "CRITICAL",
    ) -> None:
        """
        Parameters
        ----------
        cooldown_seconds  : minimum gap between emails (default 60s)
        min_tier_to_alert : only alert at this tier or above
                            "WARNING"  -> alerts on WARNING and CRITICAL
                            "CRITICAL" -> alerts on CRITICAL only
        """
        self._cooldown    = cooldown_seconds
        self._min_tier    = min_tier_to_alert
        self._last_sent   = 0.0          # epoch time of last successful send
        self._send_count  = 0            # total emails sent this session
        self._tier_rank   = {"CLEAR": 0, "CAUTION": 1,
                             "WARNING": 2, "CRITICAL": 3}

        # Load credentials from .env file
        self._sender   = None
        self._password = None
        self._receiver = None
        self._enabled  = False

        self._load_credentials()

    # ─────────────────────────────────────────────────────────────────────────
    def _load_credentials(self) -> None:
        """
        Read SMTP credentials from the .env file.

        WHY .env FILE?
          Hardcoding credentials in Python is a serious security risk.
          If the file is ever shared, uploaded to GitHub, or screenshotted,
          your email account gets compromised. The .env file:
            - Lives only on your local machine
            - Is never imported by Python (just read as text)
            - Can be excluded from git with .gitignore
        """
        if not os.path.exists(self.ENV_FILE):
            print(f"[PyroWatch Mailer] WARNING: {self.ENV_FILE} not found.")
            print(f"              Email alerts are DISABLED.")
            print(f"              Create {self.ENV_FILE} with your credentials to enable.")
            return

        creds = {}
        with open(self.ENV_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if "=" in line and not line.startswith("#"):
                    key, _, val = line.partition("=")
                    creds[key.strip()] = val.strip()

        self._sender   = creds.get("PyroWatch_EMAIL_SENDER")
        self._password = creds.get("PyroWatch_EMAIL_PASSWORD", "").replace(" ", "")
        self._receiver = creds.get("PyroWatch_EMAIL_RECEIVER", self._sender)

        if self._sender and self._password:
            self._enabled = True
            print(f"[PyroWatch Mailer] Email alerts ENABLED")
            print(f"              From : {self._sender}")
            print(f"              To   : {self._receiver}")
            print(f"              Cooldown: {self._cooldown}s between alerts")
        else:
            print(f"[PyroWatch Mailer] WARNING: Credentials incomplete in {self.ENV_FILE}")
            print(f"              Email alerts are DISABLED.")

    # ─────────────────────────────────────────────────────────────────────────
    def send_alert(
        self,
        tier       : str,
        fire_conf  : float,
        smoke_conf : float,
        risk_score : float,
        frame_num  : int,
        canvas     : np.ndarray = None,
    ) -> bool:
        """
        Send an alert email if conditions are met.

        Conditions for sending:
          1. Email is enabled (credentials loaded correctly)
          2. Tier is at or above min_tier_to_alert
          3. Cooldown period has elapsed since last email

        Parameters
        ----------
        tier       : current risk tier string
        fire_conf  : fire confidence 0.0-1.0
        smoke_conf : smoke confidence 0.0-1.0
        risk_score : weighted score W
        frame_num  : current frame number
        canvas     : HUD-rendered frame (attached as screenshot if provided)

        Returns
        -------
        bool : True if email was sent, False if skipped
        """
        # Check if alerts are enabled
        if not self._enabled:
            return False

        # Check tier threshold
        if self._tier_rank.get(tier, 0) < self._tier_rank.get(self._min_tier, 3):
            return False

        # Check cooldown
        now     = time.time()
        elapsed = now - self._last_sent
        if elapsed < self._cooldown:
            remaining = int(self._cooldown - elapsed)
            print(f"  [PyroWatch Mailer] Cooldown active -- next alert in {remaining}s")
            return False

        # All conditions met -- build and send email
        print(f"  [PyroWatch Mailer] Sending {tier} alert email...")
        success = self._send_email(
            tier       = tier,
            fire_conf  = fire_conf,
            smoke_conf = smoke_conf,
            risk_score = risk_score,
            frame_num  = frame_num,
            canvas     = canvas,
        )

        if success:
            self._last_sent  = now
            self._send_count += 1
            print(f"  [PyroWatch Mailer] Email sent successfully "
                  f"(total this session: {self._send_count})")
        return success

    # ─────────────────────────────────────────────────────────────────────────
    def _send_email(
        self,
        tier       : str,
        fire_conf  : float,
        smoke_conf : float,
        risk_score : float,
        frame_num  : int,
        canvas     : np.ndarray,
    ) -> bool:
        """
        Build the MIME email and send it via Gmail SMTP.

        HOW SMTP WORKS:
          SMTP (Simple Mail Transfer Protocol) is the standard protocol
          for sending email. Gmail's SMTP server runs on port 587 with
          STARTTLS encryption:
            1. Connect to smtp.gmail.com:587 (plain TCP)
            2. Call .starttls() to upgrade to encrypted TLS connection
            3. Login with sender email + App Password
            4. Send the MIME message
            5. Quit the connection

          MIME (Multipurpose Internet Mail Extensions) allows emails to
          carry multiple parts -- plain text, HTML, and attachments.
          We build a multipart message with:
            - A plain text body (readable in all email clients)
            - An HTML body (formatted, colour-coded)
            - A PNG image attachment (the live screenshot)
        """
        ts  = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        emoji = "🔥" if tier == "CRITICAL" else "⚠️"

        # ── Build the email message ───────────────────────────────────────
        msg = MIMEMultipart("mixed")
        msg["Subject"] = f"{emoji} {tier} HAZARD ALERT — PyroWatch Industrial Monitor"
        msg["From"]    = self._sender
        msg["To"]      = self._receiver

        # ── Plain text body ───────────────────────────────────────────────
        plain_body = f"""
PyroWatch -- Industrial Hazard Detection System
=========================================
ALERT LEVEL : {tier}
TIMESTAMP   : {ts}
FRAME       : {frame_num}

DETECTION READINGS:
  Fire Confidence  : {fire_conf:.1%}
  Smoke Confidence : {smoke_conf:.1%}
  Weighted Score   : {risk_score:.4f}

ACTION REQUIRED:
  {"EVACUATE IMMEDIATELY. Contact emergency services." if tier == "CRITICAL"
   else "Monitor situation closely. Prepare response team."}

--
PyroWatch Automated Alert System
"""
        # ── HTML body (colour-coded) ──────────────────────────────────────
        colour  = "#ff2222" if tier == "CRITICAL" else "#ff8800"
        action  = ("EVACUATE IMMEDIATELY. Contact emergency services."
                   if tier == "CRITICAL"
                   else "Monitor situation closely. Prepare response team.")

        html_body = f"""
<html><body style="font-family:monospace;background:#0a0a14;color:#dcdcdc;padding:24px;">
  <div style="border:2px solid {colour};padding:20px;border-radius:8px;max-width:600px;">
    <h2 style="color:{colour};margin:0 0 16px 0;">
      {emoji} {tier} HAZARD ALERT
    </h2>
    <table style="width:100%;border-collapse:collapse;">
      <tr><td style="color:#aaa;padding:4px 0;">Timestamp</td>
          <td style="color:#fff;padding:4px 0;">{ts}</td></tr>
      <tr><td style="color:#aaa;padding:4px 0;">Frame</td>
          <td style="color:#fff;padding:4px 0;">{frame_num}</td></tr>
      <tr><td style="color:#aaa;padding:4px 0;">Fire Confidence</td>
          <td style="color:#ff6600;padding:4px 0;font-weight:bold;">{fire_conf:.1%}</td></tr>
      <tr><td style="color:#aaa;padding:4px 0;">Smoke Confidence</td>
          <td style="color:#cccccc;padding:4px 0;font-weight:bold;">{smoke_conf:.1%}</td></tr>
      <tr><td style="color:#aaa;padding:4px 0;">Risk Score (W)</td>
          <td style="color:{colour};padding:4px 0;font-weight:bold;">{risk_score:.4f}</td></tr>
    </table>
    <div style="margin-top:16px;padding:12px;background:{colour}22;
                border-left:4px solid {colour};border-radius:4px;">
      <strong style="color:{colour};">ACTION REQUIRED:</strong>
      <p style="margin:8px 0 0 0;color:#fff;">{action}</p>
    </div>
    {"<p style='margin-top:16px;color:#888;font-size:12px;'>Screenshot attached.</p>"
     if canvas is not None else ""}
  </div>
  <p style="color:#444;font-size:11px;margin-top:12px;">
    PyroWatch Automated Alert System
  </p>
</body></html>
"""
        # Attach both plain and HTML as alternatives
        alt_part = MIMEMultipart("alternative")
        alt_part.attach(MIMEText(plain_body, "plain"))
        alt_part.attach(MIMEText(html_body,  "html"))
        msg.attach(alt_part)

        # ── Attach screenshot if canvas provided ──────────────────────────
        if canvas is not None:
            try:
                success_enc, img_buf = cv2.imencode(".png", canvas)
                if success_enc:
                    img_attachment = MIMEImage(
                        img_buf.tobytes(),
                        name=f"PyroWatch_alert_{tier}_{frame_num}.png"
                    )
                    img_attachment.add_header(
                        "Content-Disposition", "attachment",
                        filename=f"PyroWatch_alert_{tier}_{frame_num}.png"
                    )
                    msg.attach(img_attachment)
            except Exception as e:
                print(f"  [PyroWatch Mailer] Screenshot attach failed: {e}")

        # ── Send via Gmail SMTP ───────────────────────────────────────────
        try:
            with smtplib.SMTP(self.SMTP_HOST, self.SMTP_PORT, timeout=10) as server:
                server.starttls()                            # encrypt connection
                server.login(self._sender, self._password)  # authenticate
                server.sendmail(
                    self._sender,
                    self._receiver,
                    msg.as_string()
                )
            return True
        except smtplib.SMTPAuthenticationError:
            print("  [PyroWatch Mailer] ERROR: Authentication failed.")
            print("                Check your App Password in .env file.")
            print("                Make sure 2FA is enabled on your Google account.")
            return False
        except smtplib.SMTPException as e:
            print(f"  [PyroWatch Mailer] SMTP error: {e}")
            return False
        except Exception as e:
            print(f"  [PyroWatch Mailer] Unexpected error: {e}")
            return False

    # ─────────────────────────────────────────────────────────────────────────
    @property
    def enabled(self) -> bool:
        """True if credentials loaded and mailer is ready to send."""
        return self._enabled

    @property
    def send_count(self) -> int:
        """Total emails sent this session."""
        return self._send_count

    def cooldown_remaining(self) -> int:
        """Seconds until next alert can be sent. 0 if ready."""
        remaining = self._cooldown - (time.time() - self._last_sent)
        return max(0, int(remaining))



