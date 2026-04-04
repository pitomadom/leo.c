package com.ariannamethod.leo

import android.app.Activity
import android.appwidget.AppWidgetManager
import android.content.Intent
import android.os.Bundle
import android.view.ViewGroup
import android.view.inputmethod.EditorInfo
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.TextView

class LeoInputActivity : Activity() {

    private var appWidgetId = AppWidgetManager.INVALID_APPWIDGET_ID

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        appWidgetId = intent.getIntExtra(
            AppWidgetManager.EXTRA_APPWIDGET_ID,
            AppWidgetManager.INVALID_APPWIDGET_ID
        )

        val layout = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(24, 24, 24, 24)
            setBackgroundColor(0xFF0D1117.toInt())
        }

        val label = TextView(this).apply {
            text = "leo ·"
            textSize = 10f
            setTextColor(0xFF8B949E.toInt())
            setPadding(0, 0, 0, 12)
        }

        val input = EditText(this).apply {
            hint = "speak to leo..."
            textSize = 16f
            setTextColor(0xFFE6EDF3.toInt())
            setHintTextColor(0xFF484F58.toInt())
            setBackgroundColor(0xFF161B22.toInt())
            setPadding(16, 12, 16, 12)
            isSingleLine = true
            imeOptions = EditorInfo.IME_ACTION_SEND
            layoutParams = LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
            )

            setOnEditorActionListener { v, actionId, _ ->
                if (actionId == EditorInfo.IME_ACTION_SEND) {
                    val text = v.text.toString().trim().take(200)
                    if (text.isNotBlank()) submitInput(text)
                    true
                } else false
            }
        }

        layout.addView(label)
        layout.addView(input)
        setContentView(layout)

        input.requestFocus()
        window.setSoftInputMode(android.view.WindowManager.LayoutParams.SOFT_INPUT_STATE_VISIBLE)
    }

    private fun submitInput(text: String) {
        sendBroadcast(Intent(this, LeoWidget::class.java).apply {
            action = "com.ariannamethod.leo.ACTION_SUBMIT"
            putExtra("user_input", text)
        })
        setResult(RESULT_OK, Intent().apply {
            putExtra(AppWidgetManager.EXTRA_APPWIDGET_ID, appWidgetId)
        })
        finish()
    }

    override fun onBackPressed() {
        setResult(RESULT_CANCELED)
        super.onBackPressed()
    }
}
