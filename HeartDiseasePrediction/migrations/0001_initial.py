# Generated by Django 3.1.2 on 2021-08-08 23:37

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Add_Doctors',
            fields=[
                ('fname', models.CharField(max_length=100)),
                ('lname', models.CharField(max_length=100)),
                ('username', models.CharField(max_length=100)),
                ('contact_no', models.CharField(max_length=10)),
                ('doctor_email', models.EmailField(max_length=254, primary_key=True, serialize=False)),
                ('address', models.CharField(max_length=100)),
                ('province', models.CharField(max_length=100)),
                ('gender', models.CharField(max_length=6)),
                ('user_postal', models.CharField(max_length=6)),
                ('password', models.CharField(max_length=100, verbose_name='password')),
            ],
        ),
        migrations.CreateModel(
            name='Add_Tips',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('title', models.CharField(max_length=50)),
                ('description', models.TextField()),
            ],
        ),
        migrations.CreateModel(
            name='User_Admin',
            fields=[
                ('first_name', models.CharField(max_length=50)),
                ('last_name', models.CharField(max_length=50)),
                ('email', models.EmailField(max_length=254, primary_key=True, serialize=False)),
                ('password', models.CharField(max_length=100, verbose_name='password')),
            ],
        ),
        migrations.CreateModel(
            name='User_Patient',
            fields=[
                ('firstName', models.CharField(max_length=100)),
                ('lastName', models.CharField(max_length=100)),
                ('patient_email', models.EmailField(max_length=254, primary_key=True, serialize=False)),
                ('username', models.CharField(max_length=150)),
                ('gender', models.CharField(max_length=6)),
                ('address', models.CharField(max_length=150)),
                ('province', models.CharField(max_length=100)),
                ('postal_code', models.CharField(max_length=6)),
                ('phone', models.CharField(max_length=10)),
                ('password', models.CharField(max_length=100, verbose_name='password')),
            ],
        ),
        migrations.CreateModel(
            name='User_Records',
            fields=[
                ('sex', models.CharField(max_length=1)),
                ('age', models.CharField(max_length=10)),
                ('restingBP', models.CharField(max_length=10)),
                ('serum_cholestrol', models.CharField(max_length=10)),
                ('max_heart_rate', models.CharField(max_length=10)),
                ('Fluoroscopy_colored_major_vessels', models.CharField(max_length=10)),
                ('chest_pain_type', models.CharField(max_length=10)),
                ('fasting_BS_greaterthan_120', models.BooleanField(max_length=10)),
                ('peak_exercize', models.CharField(max_length=10)),
                ('restingecg', models.CharField(max_length=10)),
                ('exercise_induced_angima', models.BooleanField(max_length=10)),
                ('ST_depression', models.CharField(max_length=100)),
                ('thalium_stress', models.CharField(max_length=10)),
                ('result', models.CharField(max_length=20)),
                ('date', models.DateTimeField(auto_now=True, primary_key=True, serialize=False)),
                ('patient_email', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='HeartDiseasePrediction.user_patient')),
            ],
            options={
                'unique_together': {('patient_email', 'date')},
            },
        ),
        migrations.CreateModel(
            name='Sent_Messages',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('sent_message', models.TextField()),
                ('sent_by', models.CharField(max_length=10)),
                ('date', models.DateTimeField(auto_now=True)),
                ('doctor_email', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='HeartDiseasePrediction.add_doctors')),
                ('patient_email', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='HeartDiseasePrediction.user_patient')),
            ],
            options={
                'unique_together': {('patient_email', 'doctor_email', 'date')},
            },
        ),
        migrations.CreateModel(
            name='Feedback_Patient',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('comments', models.CharField(max_length=1000)),
                ('date', models.DateTimeField(auto_now=True)),
                ('patient_email', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='HeartDiseasePrediction.user_patient')),
            ],
            options={
                'unique_together': {('patient_email', 'date')},
            },
        ),
        migrations.CreateModel(
            name='Feedback_Doctor',
            fields=[
                ('comments', models.CharField(max_length=1000)),
                ('date', models.DateTimeField(auto_now=True, primary_key=True, serialize=False)),
                ('doctor_email', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='HeartDiseasePrediction.add_doctors')),
            ],
            options={
                'unique_together': {('doctor_email', 'date')},
            },
        ),
    ]
