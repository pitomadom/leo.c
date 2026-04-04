package com.ariannamethod.leo

import android.appwidget.AppWidgetManager
import android.content.BroadcastReceiver
import android.content.ComponentName
import android.content.Context
import android.content.Intent

class BootReceiver : BroadcastReceiver() {
    override fun onReceive(context: Context, intent: Intent) {
        if (intent.action == Intent.ACTION_BOOT_COMPLETED) {
            val ids = AppWidgetManager.getInstance(context).getAppWidgetIds(
                ComponentName(context, LeoWidget::class.java)
            )
            if (ids.isNotEmpty()) {
                context.sendBroadcast(Intent(context, LeoWidget::class.java).apply {
                    action = AppWidgetManager.ACTION_APPWIDGET_UPDATE
                    putExtra(AppWidgetManager.EXTRA_APPWIDGET_IDS, ids)
                })
            }
        }
    }
}
