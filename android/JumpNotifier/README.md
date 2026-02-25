# JumpNotifier (Android)

Simple Android app that listens for vLoad websocket events and shows a phone notification when a jump is persisted to DB.

## Event used

Server now broadcasts this websocket message after DB insert succeeds:

```json
{
  "type": "jump_saved",
  "event_id": 9,
  "jump_id": 777,
  "session_id": "20260222_195029_detect",
  "t_peak": 1771807842.008283
}
```

## Run

1. Open `android/JumpNotifier` in Android Studio.
2. Let Gradle sync.
3. Run on a phone on the same network as the vLoad server.
4. In the app, set websocket URL to `ws://<server-ip>:8080/ws`.
5. Tap **Connect**.

When a jump is detected **and saved to database**, the app logs the event and shows a local notification.
