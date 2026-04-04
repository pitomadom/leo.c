package com.ariannamethod.leo

import android.app.AlarmManager
import android.app.PendingIntent
import android.appwidget.AppWidgetManager
import android.appwidget.AppWidgetProvider
import android.content.Context
import android.content.Intent
import android.os.SystemClock
import android.util.Log
import android.widget.RemoteViews
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch

class LeoWidget : AppWidgetProvider() {

    companion object {
        private const val TAG = "LeoWidget"
        private const val ACTION_UPDATE = "com.ariannamethod.leo.ACTION_UPDATE"
        private const val ACTION_SUBMIT = "com.ariannamethod.leo.ACTION_SUBMIT"
        private const val ACTION_DREAM = "com.ariannamethod.leo.ACTION_DREAM"
        private const val UPDATE_INTERVAL = 180000L  // 3 minutes
        private const val DREAM_INTERVAL = 420000L   // 7 minutes
    }

    override fun onUpdate(context: Context, manager: AppWidgetManager, ids: IntArray) {
        for (id in ids) updateWidget(context, manager, id, null)
        scheduleUpdate(context)
        scheduleDream(context)
    }

    override fun onReceive(context: Context, intent: Intent) {
        super.onReceive(context, intent)
        val manager = AppWidgetManager.getInstance(context)
        val ids = manager.getAppWidgetIds(
            android.content.ComponentName(context, LeoWidget::class.java)
        )

        when (intent.action) {
            ACTION_UPDATE -> {
                for (id in ids) updateWidget(context, manager, id, null)
            }
            ACTION_SUBMIT -> {
                val input = intent.getStringExtra("user_input")
                for (id in ids) updateWidget(context, manager, id, input)
            }
            ACTION_DREAM -> {
                Log.d(TAG, "Dream triggered")
                for (id in ids) updateWidget(context, manager, id, null, isDream = true)
            }
        }
    }

    override fun onEnabled(context: Context) {
        scheduleUpdate(context)
        scheduleDream(context)
    }

    override fun onDisabled(context: Context) {
        cancelAlarms(context)
    }

    private fun updateWidget(
        context: Context, manager: AppWidgetManager, id: Int,
        userInput: String?, isDream: Boolean = false
    ) {
        CoroutineScope(Dispatchers.IO).launch {
            val engine = LeoEngine.getInstance(context)

            val text = when {
                userInput != null -> engine.generate(userInput)
                isDream -> engine.dream()
                else -> engine.think()
            }

            CoroutineScope(Dispatchers.Main).launch {
                val views = RemoteViews(context.packageName, R.layout.leo_widget)
                views.setTextViewText(R.id.leo_display, text)

                val clickIntent = Intent(context, LeoInputActivity::class.java).apply {
                    putExtra(AppWidgetManager.EXTRA_APPWIDGET_ID, id)
                }
                val clickPI = PendingIntent.getActivity(
                    context, id, clickIntent,
                    PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
                )
                views.setOnClickPendingIntent(R.id.leo_display, clickPI)
                manager.updateAppWidget(id, views)
            }
        }
    }

    private fun scheduleUpdate(context: Context) {
        val am = context.getSystemService(Context.ALARM_SERVICE) as AlarmManager
        val intent = Intent(context, LeoWidget::class.java).apply { action = ACTION_UPDATE }
        val pi = PendingIntent.getBroadcast(context, 0, intent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE)
        am.cancel(pi)
        am.setRepeating(AlarmManager.ELAPSED_REALTIME,
            SystemClock.elapsedRealtime() + UPDATE_INTERVAL, UPDATE_INTERVAL, pi)
    }

    private fun scheduleDream(context: Context) {
        val am = context.getSystemService(Context.ALARM_SERVICE) as AlarmManager
        val intent = Intent(context, LeoWidget::class.java).apply { action = ACTION_DREAM }
        val pi = PendingIntent.getBroadcast(context, 1, intent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE)
        am.cancel(pi)
        am.setRepeating(AlarmManager.ELAPSED_REALTIME,
            SystemClock.elapsedRealtime() + DREAM_INTERVAL, DREAM_INTERVAL, pi)
    }

    private fun cancelAlarms(context: Context) {
        val am = context.getSystemService(Context.ALARM_SERVICE) as AlarmManager
        // Update uses request code 0, Dream uses request code 1
        val codes = mapOf(ACTION_UPDATE to 0, ACTION_DREAM to 1)
        for ((action, code) in codes) {
            val intent = Intent(context, LeoWidget::class.java).apply { this.action = action }
            am.cancel(PendingIntent.getBroadcast(context, code, intent,
                PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE))
        }
    }
}
