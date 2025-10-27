# app/pdf_generator.py
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
import tempfile
import os

def get_risk_level(risk_value):
    if risk_value >= 70:
        return "Muy Alto"
    elif risk_value >= 50:
        return "Alto"
    elif risk_value >= 30:
        return "Moderado"
    else:
        return "Bajo"

def generate_pdf_report(user, patient_record, prediction):
    temp_dir = tempfile.gettempdir()
    pdf_filename = f"health_report_{prediction.id}.pdf"
    pdf_path = os.path.join(temp_dir, pdf_filename)

    doc = SimpleDocTemplate(pdf_path, pagesize=A4)
    story = []
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'], fontSize=20, textColor=colors.HexColor('#2563eb'), alignment=TA_CENTER)
    heading_style = ParagraphStyle('CustomHeading', parent=styles['Heading2'], fontSize=14, textColor=colors.HexColor('#1e40af'))

    story.append(Paragraph("Reporte de Evaluación de Riesgo de Salud", title_style))
    story.append(Spacer(1, 0.2*inch))

    # Patient info
    story.append(Paragraph(f"<b>Paciente:</b> {user.full_name} ({user.email})", styles['Normal']))
    story.append(Paragraph(f"<b>Registro:</b> {patient_record.id}", styles['Normal']))
    story.append(Spacer(1, 0.2*inch))

    # Risks table
    data = [
        ["Enfermedad", "Riesgo (%)", "Nivel"]
    ]
    for key, value in [
        ("Diabetes", prediction.diabetes_risk),
        ("Hipertensión", prediction.hypertension_risk),
        ("Cardiovascular", prediction.cardiovascular_risk),
        ("Enfermedad renal", prediction.kidney_disease_risk),
        ("Obesidad", prediction.obesity_risk),
        ("Dislipidemia", prediction.dyslipidemia_risk),
        ("Síndrome metabólico", prediction.metabolic_syndrome_risk),
    ]:
        data.append([key, f"{value}", get_risk_level(value)])

    table = Table(data, colWidths=[200, 100, 150])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#eff6ff')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.gray),
        ('ALIGN', (1,0), (-1,-1), 'CENTER'),
    ]))
    story.append(table)
    story.append(Spacer(1, 0.3*inch))

    story.append(Paragraph(f"<b>Nivel de Riesgo General:</b> {prediction.overall_risk.upper().replace('_',' ')}", styles['Normal']))
    story.append(Spacer(1, 0.2*inch))

    if prediction.recommendations:
        story.append(Paragraph("Recomendaciones:", heading_style))
        for section in ['lifestyle','medical','monitoring']:
            if prediction.recommendations.get(section):
                story.append(Paragraph(f"<b>{section.capitalize()}:</b>", styles['Normal']))
                for rec in prediction.recommendations[section]:
                    story.append(Paragraph(f"• {rec}", styles['Normal']))
                story.append(Spacer(1, 0.1*inch))

    disclaimer = "<i>Nota: Este reporte es informativo y no sustituye consejo médico profesional.</i>"
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph(disclaimer, styles['Normal']))

    doc.build(story)
    return pdf_path
