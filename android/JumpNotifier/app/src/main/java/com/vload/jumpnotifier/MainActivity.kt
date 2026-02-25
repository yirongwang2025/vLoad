package com.vload.jumpnotifier

import android.Manifest
import android.app.NotificationChannel
import android.app.NotificationManager
import android.content.Context
import android.os.Build
import android.os.Bundle
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.NotificationCompat
import androidx.core.app.NotificationManagerCompat
import org.json.JSONObject
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.Response
import okhttp3.WebSocket
import okhttp3.WebSocketListener
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.TimeUnit

class MainActivity : AppCompatActivity() {
    private lateinit var wsUrlInput: EditText
    private lateinit var connectButton: Button
    private lateinit var statusText: TextView
    private lateinit var logText: TextView

    private var ws: WebSocket? = null
    private var connected = false

    private val client = OkHttpClient.Builder()
        .readTimeout(0, TimeUnit.MILLISECONDS)
        .build()

    private val requestNotificationPermission = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { /* no-op */ }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        wsUrlInput = findViewById(R.id.wsUrlInput)
        connectButton = findViewById(R.id.connectButton)
        statusText = findViewById(R.id.statusText)
        logText = findViewById(R.id.logText)

        ensureNotificationChannel()
        ensureNotificationPermissionIfNeeded()
        updateUiState()

        connectButton.setOnClickListener {
            if (connected) disconnectWs() else connectWs()
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        disconnectWs()
        client.dispatcher.executorService.shutdown()
    }

    private fun connectWs() {
        val wsUrl = wsUrlInput.text?.toString()?.trim().orEmpty()
        if (wsUrl.isBlank()) {
            appendLog("WebSocket URL is empty.")
            return
        }
        statusText.text = getString(R.string.status_connecting)
        val request = Request.Builder().url(wsUrl).build()
        ws = client.newWebSocket(request, object : WebSocketListener() {
            override fun onOpen(webSocket: WebSocket, response: Response) {
                runOnUiThread {
                    connected = true
                    updateUiState()
                    appendLog("Connected to $wsUrl")
                }
            }

            override fun onMessage(webSocket: WebSocket, text: String) {
                handleWsMessage(text)
            }

            override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
                runOnUiThread {
                    appendLog("WebSocket error: ${t.message ?: "unknown"}")
                    connected = false
                    updateUiState()
                }
            }

            override fun onClosing(webSocket: WebSocket, code: Int, reason: String) {
                webSocket.close(1000, null)
                runOnUiThread {
                    appendLog("Closing: $code $reason")
                    connected = false
                    updateUiState()
                }
            }
        })
    }

    private fun disconnectWs() {
        ws?.close(1000, "manual disconnect")
        ws = null
        connected = false
        updateUiState()
    }

    private fun handleWsMessage(text: String) {
        try {
            val obj = JSONObject(text)
            val type = obj.optString("type", "")
            if (type == "jump_saved") {
                val jumpId = obj.optInt("jump_id", -1)
                val eventId = obj.optInt("event_id", -1)
                val sessionId = obj.optString("session_id", "")
                runOnUiThread {
                    val msg = "jump_saved: jump_id=$jumpId event_id=$eventId session=$sessionId"
                    appendLog(msg)
                    showJumpNotification(jumpId, eventId, sessionId)
                }
            }
        } catch (_: Exception) {
            // Ignore non-JSON messages.
        }
    }

    private fun showJumpNotification(jumpId: Int, eventId: Int, sessionId: String) {
        val title = "Jump saved"
        val body = "jump_id=$jumpId event_id=$eventId session=$sessionId"
        val notification = NotificationCompat.Builder(this, CHANNEL_ID)
            .setSmallIcon(android.R.drawable.ic_dialog_info)
            .setContentTitle(title)
            .setContentText(body)
            .setStyle(NotificationCompat.BigTextStyle().bigText(body))
            .setPriority(NotificationCompat.PRIORITY_HIGH)
            .setAutoCancel(true)
            .build()

        try {
            NotificationManagerCompat.from(this).notify((System.currentTimeMillis() % Int.MAX_VALUE).toInt(), notification)
        } catch (_: SecurityException) {
            appendLog("Notification permission not granted.")
        }
    }

    private fun updateUiState() {
        connectButton.text = if (connected) getString(R.string.disconnect) else getString(R.string.connect)
        statusText.text = if (connected) getString(R.string.status_connected) else getString(R.string.status_disconnected)
    }

    private fun appendLog(message: String) {
        val ts = SimpleDateFormat("HH:mm:ss", Locale.US).format(Date())
        logText.append("[$ts] $message\n")
    }

    private fun ensureNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                CHANNEL_ID,
                "Jump Notifications",
                NotificationManager.IMPORTANCE_HIGH
            )
            channel.description = "Notifications for persisted jumps."
            val nm = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
            nm.createNotificationChannel(channel)
        }
    }

    private fun ensureNotificationPermissionIfNeeded() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            requestNotificationPermission.launch(Manifest.permission.POST_NOTIFICATIONS)
        }
    }

    companion object {
        private const val CHANNEL_ID = "jump_saved_channel"
    }
}
